import sqlite3
from pathlib import Path

MIGRATIONS = [
    """
    CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS dataset_sources (
      id INTEGER PRIMARY KEY, name TEXT NOT NULL, version TEXT NOT NULL, source_url TEXT NOT NULL,
      license_status TEXT NOT NULL, license_notes TEXT NOT NULL DEFAULT '', access_notes TEXT NOT NULL DEFAULT '',
      manifest_reference TEXT, retrieval_date TEXT NOT NULL, source_reported_count INTEGER,
      UNIQUE(name, version)
    );
    CREATE TABLE IF NOT EXISTS ingestion_runs (
      id INTEGER PRIMARY KEY, dataset_source_id INTEGER NOT NULL REFERENCES dataset_sources(id),
      input_root TEXT NOT NULL, status TEXT NOT NULL, checkpoint_index INTEGER NOT NULL DEFAULT 0,
      configuration_json TEXT NOT NULL, started_at TEXT NOT NULL, finished_at TEXT, UNIQUE(dataset_source_id, input_root)
    );
    CREATE TABLE IF NOT EXISTS memes (
      id INTEGER PRIMARY KEY, dataset_source_id INTEGER NOT NULL REFERENCES dataset_sources(id),
      original_source_id TEXT NOT NULL, original_source_url TEXT, local_image_path TEXT,
      image_format TEXT, width INTEGER, height INTEGER, file_size INTEGER, sha256 TEXT, perceptual_hash TEXT,
      dataset_split TEXT, emotion_label TEXT, label_provenance TEXT, annotation_json TEXT NOT NULL DEFAULT '{}',
      license_status TEXT NOT NULL, ingestion_timestamp TEXT NOT NULL, processing_status TEXT NOT NULL,
      error_details TEXT, dataset_version TEXT NOT NULL, source_manifest_reference TEXT,
      UNIQUE(dataset_source_id, original_source_id)
    );
    CREATE INDEX IF NOT EXISTS idx_memes_sha256 ON memes(sha256);
    CREATE INDEX IF NOT EXISTS idx_memes_source_status ON memes(dataset_source_id, processing_status);
    CREATE TABLE IF NOT EXISTS perceptual_hash_bands (
      meme_id INTEGER NOT NULL REFERENCES memes(id), band TEXT NOT NULL, PRIMARY KEY(meme_id, band)
    );
    CREATE INDEX IF NOT EXISTS idx_phash_bands_band ON perceptual_hash_bands(band);
    CREATE TABLE IF NOT EXISTS duplicate_relationships (
      id INTEGER PRIMARY KEY, meme_id INTEGER NOT NULL REFERENCES memes(id), related_meme_id INTEGER NOT NULL REFERENCES memes(id),
      relation_type TEXT NOT NULL CHECK(relation_type IN ('exact', 'near')), distance INTEGER NOT NULL,
      created_at TEXT NOT NULL, UNIQUE(meme_id, related_meme_id, relation_type), CHECK(meme_id != related_meme_id)
    );
    CREATE TABLE IF NOT EXISTS ingestion_failures (
      id INTEGER PRIMARY KEY, ingestion_run_id INTEGER NOT NULL REFERENCES ingestion_runs(id), source_item_id TEXT NOT NULL,
      input_path TEXT NOT NULL, reason TEXT NOT NULL, recorded_at TEXT NOT NULL,
      UNIQUE(ingestion_run_id, source_item_id)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS multimodal_analyses (
      id INTEGER PRIMARY KEY, meme_id INTEGER NOT NULL REFERENCES memes(id), image_sha256 TEXT NOT NULL,
      pipeline_version TEXT NOT NULL, configuration_digest TEXT NOT NULL, status TEXT NOT NULL,
      ocr_text TEXT NOT NULL DEFAULT '', ocr_boxes_json TEXT NOT NULL DEFAULT '[]', ocr_confidence REAL,
      visual_caption TEXT NOT NULL DEFAULT '', combined_context TEXT NOT NULL DEFAULT '',
      ocr_provider TEXT NOT NULL, caption_provider TEXT NOT NULL, provenance_json TEXT NOT NULL DEFAULT '{}',
      error_details TEXT, warning_details TEXT, processed_at TEXT NOT NULL,
      UNIQUE(meme_id, image_sha256, pipeline_version, configuration_digest)
    );
    CREATE INDEX IF NOT EXISTS idx_analysis_meme_status ON multimodal_analyses(meme_id, status);
    """,
    """
    CREATE TABLE IF NOT EXISTS emotion_features (
      id INTEGER PRIMARY KEY, meme_id INTEGER NOT NULL REFERENCES memes(id), feature_version TEXT NOT NULL,
      text_encoder TEXT NOT NULL, image_encoder TEXT NOT NULL, text_json TEXT NOT NULL, image_json TEXT NOT NULL,
      feature_hash TEXT NOT NULL, status TEXT NOT NULL, error_details TEXT, extracted_at TEXT NOT NULL,
      UNIQUE(meme_id, feature_version)
    );
    CREATE TABLE IF NOT EXISTS emotion_models (
      id INTEGER PRIMARY KEY, model_name TEXT NOT NULL, model_version TEXT NOT NULL, artifact_path TEXT NOT NULL,
      feature_version TEXT NOT NULL, configuration_json TEXT NOT NULL, metrics_json TEXT NOT NULL DEFAULT '{}',
      trained_at TEXT NOT NULL, UNIQUE(model_name, model_version)
    );
    CREATE TABLE IF NOT EXISTS emotion_predictions (
      id INTEGER PRIMARY KEY, meme_id INTEGER NOT NULL REFERENCES memes(id), model_id INTEGER NOT NULL REFERENCES emotion_models(id),
      feature_hash TEXT NOT NULL, status TEXT NOT NULL, emotion_probabilities_json TEXT NOT NULL DEFAULT '{}',
      sentiment_probabilities_json TEXT NOT NULL DEFAULT '{}', error_details TEXT, predicted_at TEXT NOT NULL,
      UNIQUE(meme_id, model_id, feature_hash)
    );
    CREATE INDEX IF NOT EXISTS idx_emotion_predictions_meme ON emotion_predictions(meme_id, model_id);
    """,
    """
    CREATE TABLE IF NOT EXISTS source_permission_reviews (
      id INTEGER PRIMARY KEY, dataset_source_id INTEGER NOT NULL REFERENCES dataset_sources(id),
      acquisition_method TEXT NOT NULL, terms_reference TEXT NOT NULL, review_record TEXT NOT NULL,
      allowed_intended_use TEXT NOT NULL, restrictions TEXT NOT NULL, attribution TEXT NOT NULL,
      review_status TEXT NOT NULL CHECK(review_status IN ('approved', 'rejected', 'unknown')),
      reviewed_at TEXT NOT NULL, evidence_reference TEXT NOT NULL,
      UNIQUE(dataset_source_id)
    );
    CREATE TABLE IF NOT EXISTS catalog_generations (
      generation_id TEXT PRIMARY KEY, parent_generation_id TEXT REFERENCES catalog_generations(generation_id),
      parent_manifest_sha256 TEXT, catalog_identity_sha256 TEXT UNIQUE, status TEXT NOT NULL
        CHECK(status IN ('draft', 'finalized', 'retired')),
      artifact_directory TEXT, manifest_sha256 TEXT, created_at TEXT NOT NULL, finalized_at TEXT
    );
    CREATE TABLE IF NOT EXISTS catalog_generation_members (
      generation_id TEXT NOT NULL REFERENCES catalog_generations(generation_id),
      source_key TEXT NOT NULL, meme_id INTEGER NOT NULL REFERENCES memes(id), ordinal INTEGER NOT NULL,
      PRIMARY KEY(generation_id, source_key), UNIQUE(generation_id, ordinal)
    );
    CREATE TABLE IF NOT EXISTS catalog_generation_features (
      generation_id TEXT NOT NULL REFERENCES catalog_generations(generation_id), source_key TEXT NOT NULL,
      feature_kind TEXT NOT NULL CHECK(feature_kind IN ('ocr_caption', 'minilm_text', 'openclip_image', 'faiss')),
      image_sha256 TEXT NOT NULL, feature_version TEXT NOT NULL, model_provenance_sha256 TEXT NOT NULL,
      artifact_reference TEXT NOT NULL, status TEXT NOT NULL CHECK(status IN ('completed', 'failed')),
      recorded_at TEXT NOT NULL,
      PRIMARY KEY(generation_id, source_key, feature_kind, feature_version, model_provenance_sha256)
    );
    CREATE INDEX IF NOT EXISTS idx_generation_features_cache ON catalog_generation_features(
      image_sha256, feature_kind, feature_version, model_provenance_sha256, status
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS product_feedback (
      id INTEGER PRIMARY KEY,
      request_id TEXT NOT NULL,
      meme_id TEXT NOT NULL,
      selected_mood TEXT,
      retrieval_variant TEXT NOT NULL CHECK(retrieval_variant = 'a0'),
      feedback_value TEXT NOT NULL CHECK(feedback_value IN ('useful', 'not_relevant')),
      created_at TEXT NOT NULL,
      UNIQUE(request_id, meme_id)
    );
    CREATE INDEX IF NOT EXISTS idx_product_feedback_created_at ON product_feedback(created_at);
    """,
]


def connect(database: Path) -> sqlite3.Connection:
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA busy_timeout = 30000")
    return connection


def migrate(connection: sqlite3.Connection) -> None:
    connection.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
    )
    current = {
        row["version"] for row in connection.execute("SELECT version FROM schema_migrations")
    }
    for version, sql in enumerate(MIGRATIONS, start=1):
        if version not in current:
            connection.executescript(sql)
            connection.execute(
                "INSERT INTO schema_migrations(version, applied_at) VALUES (?, datetime('now'))",
                (version,),
            )
    connection.commit()
