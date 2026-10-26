import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from app.datasets.catalog_generations import (
    CatalogGenerationManager,
    CatalogPlanner,
    PermissionReview,
)
from app.datasets.ingestion import DatasetIngester, IngestionOptions, SourceDetails


def make_image(path: Path, color: str, text: str = "", offset: int = 10) -> None:
    image = Image.new("RGB", (128, 128), color)
    ImageDraw.Draw(image).text((offset, 10), text, fill="white")
    image.save(path)


@pytest.fixture
def approved_review() -> PermissionReview:
    return PermissionReview(
        acquisition_method="synthetic fixture",
        terms_reference="https://example.test/terms",
        review_record="test-review-1",
        allowed_intended_use="development testing",
        restrictions="no external distribution",
        attribution="Synthetic fixture",
        review_status="approved",
        reviewed_at="2026-10-26",
        evidence_reference="tests/test_catalog_generations.py",
    )


def test_dry_run_is_deterministic_and_records_fixture_failures(
    tmp_path: Path, approved_review: PermissionReview
) -> None:
    root = tmp_path / "synthetic"
    root.mkdir()
    make_image(root / "valid.png", "red", "valid")
    (root / "exact.png").write_bytes((root / "valid.png").read_bytes())
    make_image(root / "near.png", "red", "valid", offset=11)
    (root / "corrupt.png").write_text("not an image", encoding="utf-8")
    planner = CatalogPlanner()
    expected = ("valid.png", "exact.png", "near.png", "corrupt.png", "missing.png")
    first = planner.dry_run(root, approved_review, near_duplicate_distance=64, expected_source_keys=expected)
    second = planner.dry_run(root, approved_review, near_duplicate_distance=64, expected_source_keys=expected)
    assert [item.source_key for item in first.accepted] == [item.source_key for item in second.accepted]
    assert {item.source_key for item in first.rejected} == {"corrupt.png", "missing.png"}
    assert first.exact_duplicates == (("valid.png", "exact.png"),)
    assert first.near_duplicates


def test_dry_run_rejects_unknown_or_rejected_permission(tmp_path: Path) -> None:
    root = tmp_path / "synthetic"
    root.mkdir()
    make_image(root / "valid.png", "red")
    rejected = PermissionReview(
        "synthetic", "terms", "review", "testing", "restriction", "attribution", "rejected", "today", "evidence"
    )
    with pytest.raises(ValueError, match="approved"):
        CatalogPlanner().dry_run(root, rejected)


def _import_synthetic_members(tmp_path: Path) -> tuple[DatasetIngester, list[int], int]:
    root = tmp_path / "images"
    root.mkdir()
    make_image(root / "two.png", "blue", "two")
    make_image(root / "one.png", "red", "one")
    ingester = DatasetIngester(tmp_path / "metadata.db")
    source = SourceDetails("synthetic", "v1", "https://example.test/source", "pending")
    ingester.ingest(root, source, IngestionOptions(batch_size=1))
    source_id = ingester.connection.execute("SELECT id FROM dataset_sources").fetchone()[0]
    ids = [row[0] for row in ingester.connection.execute("SELECT id FROM memes ORDER BY id")]
    return ingester, ids, source_id


def test_permission_gate_blocks_generation_promotion(tmp_path: Path, approved_review: PermissionReview) -> None:
    ingester, ids, source_id = _import_synthetic_members(tmp_path)
    manager = CatalogGenerationManager(tmp_path / "metadata.db", tmp_path / "generations")
    with pytest.raises(ValueError, match="permission review blocks"):
        manager.begin("blocked", ids)
    rejected = PermissionReview(**{**approved_review.__dict__, "review_status": "rejected"})
    manager.record_permission_review(source_id, rejected)
    with pytest.raises(ValueError, match="permission review blocks"):
        manager.begin("still-blocked", ids)
    manager.close()
    ingester.close()


def test_generation_finalization_parent_and_rollback(tmp_path: Path, approved_review: PermissionReview) -> None:
    ingester, ids, source_id = _import_synthetic_members(tmp_path)
    root = tmp_path / "generations"
    manager = CatalogGenerationManager(tmp_path / "metadata.db", root)
    manager.record_permission_review(source_id, approved_review)
    manager.begin("generation-1", ids)
    first = manager.finalize("generation-1")
    assert (first / "manifest.json").is_file()
    manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["members"]) == 2
    manager.begin("generation-2", ids, parent_generation_id="generation-1")
    second = manager.finalize("generation-2")
    child = json.loads((second / "manifest.json").read_text(encoding="utf-8"))
    assert child["parent_generation_id"] == "generation-1"
    selector = manager.rollback("generation-1")
    assert json.loads(selector.read_text(encoding="utf-8"))["generation_id"] == "generation-1"
    manager.close()
    ingester.close()


def test_finalization_failure_leaves_no_partial_artifact(
    tmp_path: Path, approved_review: PermissionReview, monkeypatch: pytest.MonkeyPatch
) -> None:
    ingester, ids, source_id = _import_synthetic_members(tmp_path)
    root = tmp_path / "generations"
    manager = CatalogGenerationManager(tmp_path / "metadata.db", root)
    manager.record_permission_review(source_id, approved_review)
    manager.begin("atomic", ids)

    def fail_replace(_: Path, __: Path) -> None:
        raise OSError("simulated interruption")

    monkeypatch.setattr("app.datasets.catalog_generations.os.replace", fail_replace)
    with pytest.raises(OSError, match="simulated interruption"):
        manager.finalize("atomic")
    assert not (root / "atomic").exists()
    assert not (root / ".atomic.staging").exists()
    manager.close()
    ingester.close()


def test_resume_and_embedding_version_isolation(tmp_path: Path, approved_review: PermissionReview) -> None:
    root = tmp_path / "images"
    root.mkdir()
    make_image(root / "first.png", "red")
    make_image(root / "second.png", "blue")
    database = tmp_path / "metadata.db"
    ingester = DatasetIngester(database)
    source = SourceDetails("synthetic", "v1", "https://example.test/source", "pending")
    first = ingester.ingest(root, source, IngestionOptions(batch_size=1, max_images=1))
    assert first.accepted == 1
    ingester.connection.execute("UPDATE ingestion_runs SET status = 'running' WHERE id = ?", (first.run_id,))
    ingester.connection.commit()
    resumed = ingester.ingest(root, source, IngestionOptions(batch_size=1))
    assert resumed.accepted == 1
    source_id = ingester.connection.execute("SELECT id FROM dataset_sources").fetchone()[0]
    ids = [row[0] for row in ingester.connection.execute("SELECT id FROM memes ORDER BY id")]
    manager = CatalogGenerationManager(database, tmp_path / "generations")
    manager.record_permission_review(source_id, approved_review)
    manager.begin("features", ids)
    keys = [row[0] for row in manager.connection.execute(
        "SELECT source_key FROM catalog_generation_members ORDER BY ordinal"
    )]
    manager.record_feature("features", keys[0], "minilm_text", "minilm-v1", "model-a", "memory://one")
    manager.record_feature("features", keys[1], "minilm_text", "minilm-v2", "model-b", "memory://two")
    with pytest.raises(ValueError, match="incompatible"):
        manager.assert_feature_isolation("features", "minilm_text")
    manager.close()
    ingester.close()
