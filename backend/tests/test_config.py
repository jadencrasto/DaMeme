from app.core.config import Settings


def test_settings_parse_explicit_cors_origins() -> None:
    settings = Settings(cors_origins="http://localhost:5173, http://127.0.0.1:5173")
    assert settings.allowed_origins == ["http://localhost:5173", "http://127.0.0.1:5173"]
