from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


@dataclass(frozen=True)
class CatalogPaths:
    catalog_id: str
    index_dir: Path
    metadata_csv: Path
    image_root: Path | None
    expected_count: int
    expected_dimensions: tuple[int, int, int] | None = (384, 768, 1152)
    export_records: Path | None = None
    export_metadata: Path | None = None


class Settings(BaseSettings):
    """Runtime configuration read from environment variables or a local .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="DAMEME_",
        case_sensitive=False,
        extra="ignore",
    )

    environment: str = "development"
    api_v1_prefix: str = "/api/v1"
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    database_url: str = "sqlite:///./dameme.db"
    recommendation_index_dir: Path = Path("../data/retrieval/memematch_281_indexes")
    recommendation_metadata_csv: Path = Path(
        "../data/retrieval/memematch_281/memematch_281_metadata.csv"
    )
    recommendation_image_root: Path | None = None
    recommendation_catalog: str = "memematch_281"
    recommendation_catalog_root: Path = Path("../data/retrieval/catalogs")
    recommendation_cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    recommendation_request_timeout_seconds: float = 15.0
    expanded_emotion_text_model: Path | None = None
    phase5_bundle_dir: Path | None = None

    def active_catalog(self) -> CatalogPaths:
        """Resolve only known catalog layouts; the legacy catalog remains default."""
        if self.recommendation_catalog == "memematch_281":
            export_root = Path("../data/retrieval/memematch_281/memematch_281")
            return CatalogPaths(
                "memematch_281",
                self.recommendation_index_dir,
                self.recommendation_metadata_csv,
                self.recommendation_image_root,
                281,
                export_records=export_root.with_suffix(".records.json"),
                export_metadata=export_root.with_suffix(".metadata.json"),
            )
        if self.recommendation_catalog == "memematch_1000_v2":
            root = self.recommendation_catalog_root / "memematch_1000_v2"
            return CatalogPaths(
                "memematch_1000_v2",
                root / "indexes",
                root / "memematch_1000_metadata_v2.csv",
                root,
                1000,
                export_records=root / "exports" / "memematch_1000_v2.records.json",
                export_metadata=root / "exports" / "memematch_1000_v2.metadata.json",
            )
        if self.recommendation_catalog == "dameme_5500_v1":
            root = self.recommendation_catalog_root / "dameme_5500_v1"
            return CatalogPaths(
                "dameme_5500_v1",
                root / "indexes",
                root / "dameme_5500_metadata.csv",
                root,
                5500,
                export_records=root / "exports" / "dameme_5500_v1.records.json",
                export_metadata=root / "exports" / "dameme_5500_v1.metadata.json",
            )
        raise ValueError(
            "recommendation_catalog must be memematch_281, memematch_1000_v2, or dameme_5500_v1"
        )

    @property
    def allowed_origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
