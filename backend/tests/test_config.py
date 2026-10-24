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
