"""Immutable, permission-gated catalog generation support.

This module deliberately writes only a caller-selected generation root.  It never changes
the active application catalog or existing retrieval artifacts.
"""

import hashlib
import json
import os
import shutil
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from app.datasets.database import connect, migrate
from app.datasets.hashing import hamming_distance, perceptual_hash, sha256_file
from app.datasets.validation import ImageValidationError, validate_image


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class PermissionReview:
    acquisition_method: str
    terms_reference: str
    review_record: str
    allowed_intended_use: str
    restrictions: str
    attribution: str
    review_status: str
    reviewed_at: str
    evidence_reference: str

    def validate_for_promotion(self) -> None:
        required = asdict(self)
        missing = [name for name, value in required.items() if not value.strip()]
        if missing:
            raise ValueError(f"permission review missing required fields: {', '.join(missing)}")
        if self.review_status != "approved":
            raise ValueError("permission review must be approved before catalog promotion")


@dataclass(frozen=True)
class PlannedItem:
    source_key: str
    input_path: str
    sha256: str
    perceptual_hash: str
    status: str
    reason: str = ""


@dataclass(frozen=True)
class DryRunResult:
    accepted: tuple[PlannedItem, ...]
    rejected: tuple[PlannedItem, ...]
    exact_duplicates: tuple[tuple[str, str], ...]
    near_duplicates: tuple[tuple[str, str], ...]


class CatalogPlanner:
    """Read-only deterministic validation plan for a prospective source directory."""

    def dry_run(
        self,
        root: Path,
        review: PermissionReview,
        *,
        max_bytes: int = 20 * 1024 * 1024,
        max_pixels: int = 40_000_000,
        near_duplicate_distance: int = 6,
        expected_source_keys: tuple[str, ...] = (),
    ) -> DryRunResult:
        review.validate_for_promotion()
        root = root.resolve(strict=True)
        if not root.is_dir():
            raise ValueError("input root must be a directory")
        accepted: list[PlannedItem] = []
        rejected: list[PlannedItem] = []
        exact: list[tuple[str, str]] = []
        near: list[tuple[str, str]] = []
        hashes: dict[str, str] = {}
        prior: list[PlannedItem] = []
        for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
            source_key = path.relative_to(root).as_posix()
            try:
                validate_image(path, max_bytes=max_bytes, max_pixels=max_pixels)
                item = PlannedItem(source_key, str(path), sha256_file(path), perceptual_hash(path), "accepted")
            except (ImageValidationError, OSError, ValueError) as exc:
                rejected.append(PlannedItem(source_key, str(path), "", "", "rejected", str(exc)))
                continue
            if item.sha256 in hashes:
                exact.append((source_key, hashes[item.sha256]))
            for previous in prior:
                if hamming_distance(item.perceptual_hash, previous.perceptual_hash) <= near_duplicate_distance:
                    if item.sha256 != previous.sha256:
                        near.append((source_key, previous.source_key))
            hashes[item.sha256] = source_key
            prior.append(item)
            accepted.append(item)
        found = {item.source_key for item in accepted} | {item.source_key for item in rejected}
        for source_key in sorted(set(expected_source_keys) - found):
            rejected.append(
                PlannedItem(source_key, str(root / source_key), "", "", "rejected", "file is missing")
            )
        return DryRunResult(tuple(accepted), tuple(rejected), tuple(exact), tuple(near))


class CatalogGenerationManager:
    """Builds independently addressable catalog generations from approved imported records."""

    def __init__(self, database: Path, artifact_root: Path):
        self.connection = connect(database)
        migrate(self.connection)
        self.artifact_root = artifact_root.resolve()
        self.artifact_root.mkdir(parents=True, exist_ok=True)

    def close(self) -> None:
        self.connection.close()

    def record_permission_review(self, dataset_source_id: int, review: PermissionReview) -> None:
        if review.review_status == "approved":
            review.validate_for_promotion()
        with self.connection:
            self.connection.execute(
                """INSERT INTO source_permission_reviews(dataset_source_id, acquisition_method, terms_reference,
                review_record, allowed_intended_use, restrictions, attribution, review_status, reviewed_at,
                evidence_reference) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(dataset_source_id) DO UPDATE SET acquisition_method=excluded.acquisition_method,
                terms_reference=excluded.terms_reference, review_record=excluded.review_record,
                allowed_intended_use=excluded.allowed_intended_use, restrictions=excluded.restrictions,
                attribution=excluded.attribution, review_status=excluded.review_status,
                reviewed_at=excluded.reviewed_at, evidence_reference=excluded.evidence_reference""",
                (dataset_source_id, *asdict(review).values()),
            )

    def begin(self, generation_id: str, member_ids: list[int], parent_generation_id: str | None = None) -> None:
        if not generation_id or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for character in generation_id):
            raise ValueError("generation ID must use lowercase letters, digits, hyphens, or underscores")
        if not member_ids:
            raise ValueError("catalog generation requires at least one member")
        if len(set(member_ids)) != len(member_ids):
            raise ValueError("catalog generation member IDs must be unique")
        existing = self.connection.execute(
            "SELECT 1 FROM catalog_generations WHERE generation_id = ?", (generation_id,)
        ).fetchone()
        if existing or (self.artifact_root / generation_id).exists():
            raise ValueError(f"generation already exists: {generation_id}")
        parent_hash = None
        if parent_generation_id:
            parent = self.connection.execute(
                "SELECT manifest_sha256, status FROM catalog_generations WHERE generation_id = ?",
                (parent_generation_id,),
            ).fetchone()
            if not parent or parent["status"] != "finalized":
                raise ValueError("parent generation must be finalized")
            parent_hash = parent["manifest_sha256"]
        rows = self._approved_members(member_ids)
        source_keys = [self._source_key(row) for row in rows]
        if len(set(source_keys)) != len(source_keys):
            raise ValueError("catalog generation source keys must be unique")
        with self.connection:
            self.connection.execute(
                "INSERT INTO catalog_generations(generation_id, parent_generation_id, parent_manifest_sha256, status, created_at) VALUES (?, ?, ?, 'draft', ?)",
                (generation_id, parent_generation_id, parent_hash, utc_now()),
            )
            self.connection.executemany(
                "INSERT INTO catalog_generation_members(generation_id, source_key, meme_id, ordinal) VALUES (?, ?, ?, ?)",
                [(generation_id, key, row["id"], ordinal) for ordinal, (key, row) in enumerate(zip(source_keys, rows, strict=True))],
            )

    def record_feature(
        self,
        generation_id: str,
        source_key: str,
        feature_kind: str,
        feature_version: str,
        model_provenance_sha256: str,
        artifact_reference: str,
        *,
        status: str = "completed",
    ) -> None:
        if status not in {"completed", "failed"}:
            raise ValueError("feature status must be completed or failed")
        row = self.connection.execute(
            """SELECT memes.sha256 FROM catalog_generation_members JOIN memes ON memes.id = meme_id
            WHERE generation_id = ? AND source_key = ?""",
            (generation_id, source_key),
        ).fetchone()
        if not row:
            raise ValueError("feature source key is not a member of this generation")
        with self.connection:
            self.connection.execute(
                """INSERT INTO catalog_generation_features(generation_id, source_key, feature_kind, image_sha256,
                feature_version, model_provenance_sha256, artifact_reference, status, recorded_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (generation_id, source_key, feature_kind, row["sha256"], feature_version,
                 model_provenance_sha256, artifact_reference, status, utc_now()),
            )

    def assert_feature_isolation(self, generation_id: str, feature_kind: str) -> None:
        rows = self.connection.execute(
            """SELECT feature_version, model_provenance_sha256 FROM catalog_generation_features
            WHERE generation_id = ? AND feature_kind = ? AND status = 'completed'""",
            (generation_id, feature_kind),
        ).fetchall()
        versions = {(row["feature_version"], row["model_provenance_sha256"]) for row in rows}
        if len(versions) > 1:
            raise ValueError(f"incompatible {feature_kind} versions in generation {generation_id}")

    def finalize(self, generation_id: str) -> Path:
        generation = self.connection.execute(
            "SELECT * FROM catalog_generations WHERE generation_id = ?", (generation_id,)
        ).fetchone()
        if not generation or generation["status"] != "draft":
            raise ValueError("only an existing draft generation can be finalized")
        members = self.connection.execute(
            """SELECT catalog_generation_members.source_key, catalog_generation_members.ordinal, memes.sha256,
            memes.original_source_id, dataset_sources.name, dataset_sources.version
            FROM catalog_generation_members JOIN memes ON memes.id = meme_id
            JOIN dataset_sources ON dataset_sources.id = memes.dataset_source_id
            WHERE generation_id = ? ORDER BY ordinal""",
            (generation_id,),
        ).fetchall()
        if not members or len({row["source_key"] for row in members}) != len(members):
            raise ValueError("generation membership is empty or has duplicate source keys")
        payload_members = [dict(row) for row in members]
        manifest = {
            "format_version": 1,
            "generation_id": generation_id,
            "parent_generation_id": generation["parent_generation_id"],
            "parent_manifest_sha256": generation["parent_manifest_sha256"],
            "members": payload_members,
        }
        identity = _canonical_digest(manifest)
        manifest["catalog_identity_sha256"] = identity
        final_dir = self.artifact_root / generation_id
        staging_dir = self.artifact_root / f".{generation_id}.staging"
        if final_dir.exists() or staging_dir.exists():
            raise ValueError("generation artifact directory already exists")
        try:
            staging_dir.mkdir()
            (staging_dir / "source_keys.json").write_text(
                json.dumps([row["source_key"] for row in members], indent=2), encoding="utf-8"
            )
            manifest_path = staging_dir / "manifest.json"
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            manifest_hash = _file_digest(manifest_path)
            os.replace(staging_dir, final_dir)
        except BaseException:
            shutil.rmtree(staging_dir, ignore_errors=True)
            raise
        with self.connection:
            self.connection.execute(
                """UPDATE catalog_generations SET status = 'finalized', artifact_directory = ?,
                catalog_identity_sha256 = ?, manifest_sha256 = ?, finalized_at = ? WHERE generation_id = ?""",
                (str(final_dir), identity, manifest_hash, utc_now(), generation_id),
            )
        return final_dir

    def rollback(self, generation_id: str) -> Path:
        row = self.connection.execute(
            "SELECT artifact_directory, status, manifest_sha256 FROM catalog_generations WHERE generation_id = ?",
            (generation_id,),
        ).fetchone()
        if not row or row["status"] != "finalized" or not row["artifact_directory"]:
            raise ValueError("rollback target must be a finalized generation")
        target = Path(row["artifact_directory"])
        manifest = target / "manifest.json"
        if not manifest.is_file() or _file_digest(manifest) != row["manifest_sha256"]:
            raise ValueError("rollback target manifest is missing or has changed")
        temporary = self.artifact_root / ".active-generation.tmp"
        selector = self.artifact_root / "active-generation.json"
        temporary.write_text(
            json.dumps({"generation_id": generation_id, "manifest_sha256": row["manifest_sha256"]}, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, selector)
        return selector

    def _approved_members(self, member_ids: list[int]):
        placeholders = ",".join("?" for _ in member_ids)
        rows = self.connection.execute(
            f"""SELECT memes.id, memes.original_source_id, dataset_sources.name, dataset_sources.version,
            dataset_sources.source_url, source_permission_reviews.review_status FROM memes
            JOIN dataset_sources ON dataset_sources.id = memes.dataset_source_id
            LEFT JOIN source_permission_reviews ON source_permission_reviews.dataset_source_id = dataset_sources.id
            WHERE memes.id IN ({placeholders}) ORDER BY dataset_sources.name, dataset_sources.version,
            memes.original_source_id""",
            member_ids,
        ).fetchall()
        if len(rows) != len(member_ids):
            raise ValueError("generation includes unknown member IDs")
        blocked = [row["id"] for row in rows if row["review_status"] != "approved"]
        if blocked:
            raise ValueError(f"permission review blocks promotion for meme IDs: {blocked}")
        missing_origin = [row["id"] for row in rows if not row["source_url"].strip()]
        if missing_origin:
            raise ValueError(f"source URL blocks promotion for meme IDs: {missing_origin}")
        return rows

    @staticmethod
    def _source_key(row: object) -> str:
        return f"{row['name']}:{row['version']}:{row['original_source_id']}"
