import pytest

from app.core.config import Settings


def test_settings_parse_explicit_cors_origins() -> None:
    settings = Settings(cors_origins="http://localhost:5173, http://127.0.0.1:5173")
    assert settings.allowed_origins == ["http://localhost:5173", "http://127.0.0.1:5173"]


def test_catalog_selection_defaults_to_281_and_has_an_isolated_v2_layout() -> None:
    assert Settings().active_catalog().catalog_id == "memematch_281"
    catalog = Settings(recommendation_catalog="memematch_1000_v2").active_catalog()
    assert catalog.expected_count == 1000
    assert str(catalog.index_dir).endswith("catalogs\\memematch_1000_v2\\indexes")
    with pytest.raises(ValueError, match="recommendation_catalog"):
        Settings(recommendation_catalog="unknown").active_catalog()


def test_legacy_catalog_reuses_phase5_media_root_when_no_legacy_root_is_configured(
    tmp_path,
) -> None:
    phase5_images = tmp_path / "phase5-images"
    settings = Settings(
        recommendation_image_root=None,
        phase5_media_root=phase5_images,
    )
    assert settings.active_catalog().image_root == phase5_images


def test_explicit_legacy_image_root_takes_precedence_over_phase5_media_root(tmp_path) -> None:
    legacy_images, phase5_images = tmp_path / "legacy", tmp_path / "phase5"
    settings = Settings(
        recommendation_image_root=legacy_images,
        phase5_media_root=phase5_images,
    )
    assert settings.active_catalog().image_root == legacy_images
