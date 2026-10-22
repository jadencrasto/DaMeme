import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from app.datasets.database import connect, migrate
from app.datasets.hashing import hamming_distance, hash_bands, perceptual_hash, sha256_file
from app.datasets.validation import ImageValidationError, validate_image


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class SourceDetails:
    name: str
    version: str
    source_url: str
    license_status: str
    license_notes: str = ""
    access_notes: str = ""
    manifest_reference: str | None = None
    source_reported_count: int | None = None


@dataclass(frozen=True)
class IngestionOptions:
    batch_size: int = 100
    max_images: int | None = None
    max_bytes: int = 20 * 1024 * 1024
    max_pixels: int = 40_000_000
    near_duplicate_distance: int = 6


@dataclass(frozen=True)
class IngestionResult:
    run_id: int
    scanned: int
    accepted: int
    rejected: int
    skipped: int


def _metadata_for(
    path: Path, root: Path, metadata: dict[str, dict[str, object]]
) -> tuple[str, dict[str, object]]:
    source_id = path.relative_to(root).as_posix()
    return source_id, metadata.get(source_id, {})


class DatasetIngester:
    def __init__(self, database: Path):
        self.connection = connect(database)
        migrate(self.connection)

    def close(self) -> None:
        self.connection.close()

    def ingest(
        self,
        root: Path,
        source: SourceDetails,
        options: IngestionOptions,
        metadata: dict[str, dict[str, object]] | None = None,
    ) -> IngestionResult:
        root = root.resolve(strict=True)
        if not root.is_dir():
            raise ValueError("input root must be a directory")
        if options.batch_size < 1:
            raise ValueError("batch size must be positive")
        metadata = metadata or {}
        source_id = self._ensure_source(source)
        run_id = self._start_or_resume_run(source_id, root, options)
        files = sorted(path for path in root.rglob("*") if path.is_file())
        if options.max_images is not None:
            files = files[: options.max_images]
        start_index = self.connection.execute(
            "SELECT checkpoint_index FROM ingestion_runs WHERE id = ?", (run_id,)
        ).fetchone()[0]
        accepted = rejected = skipped = 0
        for index, path in enumerate(files[start_index:], start=start_index):
            try:
                outcome = self._ingest_one(path, root, source_id, source, run_id, options, metadata)
                accepted += outcome == "accepted"
                skipped += outcome == "skipped"
            except (ImageValidationError, OSError, ValueError) as exc:
                rejected += 1
                with self.connection:
                    self.connection.execute(
                        "INSERT OR REPLACE INTO ingestion_failures(ingestion_run_id, source_item_id, input_path, reason, recorded_at) VALUES (?, ?, ?, ?, ?)",
                        (run_id, path.relative_to(root).as_posix(), str(path), str(exc), utc_now()),
                    )
            if (index + 1) % options.batch_size == 0:
                self._checkpoint(run_id, index + 1)
        self._checkpoint(run_id, len(files), status="completed")
        return IngestionResult(run_id, len(files), accepted, rejected, skipped)

    def _ensure_source(self, source: SourceDetails) -> int:
        with self.connection:
            self.connection.execute(
                """INSERT INTO dataset_sources(name, version, source_url, license_status, license_notes, access_notes, manifest_reference, retrieval_date, source_reported_count) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(name, version) DO UPDATE SET source_url=excluded.source_url, license_status=excluded.license_status, license_notes=excluded.license_notes, access_notes=excluded.access_notes, manifest_reference=excluded.manifest_reference, source_reported_count=excluded.source_reported_count""",
                (
                    source.name,
                    source.version,
                    source.source_url,
                    source.license_status,
                    source.license_notes,
                    source.access_notes,
                    source.manifest_reference,
                    utc_now(),
                    source.source_reported_count,
                ),
            )
        return self.connection.execute(
            "SELECT id FROM dataset_sources WHERE name = ? AND version = ?",
            (source.name, source.version),
        ).fetchone()[0]

    def _start_or_resume_run(self, source_id: int, root: Path, options: IngestionOptions) -> int:
        row = self.connection.execute(
            "SELECT id FROM ingestion_runs WHERE dataset_source_id = ? AND input_root = ?",
            (source_id, str(root)),
        ).fetchone()
        if row:
            with self.connection:
                self.connection.execute(
                    "UPDATE ingestion_runs SET status = 'running', finished_at = NULL WHERE id = ?",
                    (row[0],),
                )
            return row[0]
        with self.connection:
            cursor = self.connection.execute(
                "INSERT INTO ingestion_runs(dataset_source_id, input_root, status, configuration_json, started_at) VALUES (?, ?, 'running', ?, ?)",
                (source_id, str(root), json.dumps(options.__dict__, sort_keys=True), utc_now()),
            )
        return cursor.lastrowid

    def _checkpoint(self, run_id: int, index: int, status: str = "running") -> None:
        with self.connection:
            self.connection.execute(
                "UPDATE ingestion_runs SET checkpoint_index = ?, status = ?, finished_at = CASE WHEN ? = 'completed' THEN ? ELSE finished_at END WHERE id = ?",
                (index, status, status, utc_now(), run_id),
            )

    def _ingest_one(
        self,
        path: Path,
        root: Path,
        dataset_source_id: int,
        source: SourceDetails,
        run_id: int,
        options: IngestionOptions,
        metadata: dict[str, dict[str, object]],
    ) -> str:
        resolved = path.resolve(strict=True)
        if root not in resolved.parents:
            raise ValueError("candidate path escapes input root")
        original_id, fields = _metadata_for(resolved, root, metadata)
        existing = self.connection.execute(
            "SELECT id FROM memes WHERE dataset_source_id = ? AND original_source_id = ?",
            (dataset_source_id, original_id),
        ).fetchone()
        if existing:
            return "skipped"
        info = validate_image(resolved, max_bytes=options.max_bytes, max_pixels=options.max_pixels)
        sha256, phash = sha256_file(resolved), perceptual_hash(resolved)
        with self.connection:
            cursor = self.connection.execute(
                """INSERT INTO memes(dataset_source_id, original_source_id, original_source_url, local_image_path, image_format, width, height, file_size, sha256, perceptual_hash, dataset_split, emotion_label, label_provenance, annotation_json, license_status, ingestion_timestamp, processing_status, dataset_version, source_manifest_reference) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'validated', ?, ?)""",
                (
                    dataset_source_id,
                    original_id,
                    fields.get("source_url"),
                    str(resolved),
                    info.image_format,
                    info.width,
                    info.height,
                    info.file_size,
                    sha256,
                    phash,
                    fields.get("split"),
                    fields.get("emotion_label"),
                    fields.get("label_provenance"),
                    json.dumps(fields.get("annotations", {}), sort_keys=True),
                    source.license_status,
                    utc_now(),
                    source.version,
                    source.manifest_reference,
                ),
            )
            meme_id = cursor.lastrowid
            exact = self.connection.execute(
                "SELECT id FROM memes WHERE sha256 = ? AND id != ?", (sha256, meme_id)
            ).fetchall()
            for row in exact:
                self._record_duplicate(meme_id, row[0], "exact", 0)
                self.connection.execute(
                    "UPDATE memes SET processing_status = 'duplicate_exact' WHERE id = ?",
                    (meme_id,),
                )
            candidates = self.connection.execute(
                "SELECT DISTINCT meme_id, perceptual_hash, sha256 FROM perceptual_hash_bands JOIN memes ON memes.id = meme_id WHERE band IN ({}) AND meme_id != ?".format(
                    ",".join("?" * len(hash_bands(phash)))
                ),
                (*hash_bands(phash), meme_id),
            ).fetchall()
            for candidate in candidates:
                distance = hamming_distance(phash, candidate["perceptual_hash"])
                if candidate["sha256"] != sha256 and distance <= options.near_duplicate_distance:
                    self._record_duplicate(meme_id, candidate["meme_id"], "near", distance)
            self.connection.executemany(
                "INSERT INTO perceptual_hash_bands(meme_id, band) VALUES (?, ?)",
                [(meme_id, band) for band in hash_bands(phash)],
            )
        return "accepted"

    def _record_duplicate(
        self, meme_id: int, related_id: int, relation_type: str, distance: int
    ) -> None:
        first, second = sorted((meme_id, related_id))
        self.connection.execute(
            "INSERT OR IGNORE INTO duplicate_relationships(meme_id, related_meme_id, relation_type, distance, created_at) VALUES (?, ?, ?, ?, ?)",
            (first, second, relation_type, distance, utc_now()),
        )

    def statistics(self) -> dict[str, object]:
        counts = self.connection.execute(
            "SELECT COUNT(*) AS total, SUM(processing_status = 'validated') AS valid, SUM(processing_status = 'duplicate_exact') AS exact_duplicate FROM memes"
        ).fetchone()
        by_dataset = [
            dict(row)
            for row in self.connection.execute(
                "SELECT name, version, COUNT(memes.id) AS count FROM dataset_sources LEFT JOIN memes ON dataset_sources.id = memes.dataset_source_id GROUP BY dataset_sources.id"
            )
        ]
        return {
            "total_records": counts["total"],
            "valid_images": counts["valid"] or 0,
            "exact_duplicates": counts["exact_duplicate"] or 0,
            "near_duplicates": self.connection.execute(
                "SELECT COUNT(*) FROM duplicate_relationships WHERE relation_type = 'near'"
            ).fetchone()[0],
            "failures": self.connection.execute(
                "SELECT COUNT(*) FROM ingestion_failures"
            ).fetchone()[0],
            "by_dataset": by_dataset,
            "formats": [
                dict(row)
                for row in self.connection.execute(
                    "SELECT image_format, COUNT(*) AS count FROM memes GROUP BY image_format"
                )
            ],
            "splits": [
                dict(row)
                for row in self.connection.execute(
                    "SELECT dataset_split, COUNT(*) AS count FROM memes WHERE dataset_split IS NOT NULL GROUP BY dataset_split"
                )
            ],
            "emotion_labels": [
                dict(row)
                for row in self.connection.execute(
                    "SELECT emotion_label, COUNT(*) AS count FROM memes WHERE emotion_label IS NOT NULL GROUP BY emotion_label"
                )
            ],
            "storage_bytes": self.connection.execute(
                "SELECT COALESCE(SUM(file_size), 0) FROM memes"
            ).fetchone()[0],
        }

    def export_manifest(self, dataset_name: str) -> dict[str, object]:
        source = self.connection.execute(
            "SELECT * FROM dataset_sources WHERE name = ? ORDER BY id DESC LIMIT 1", (dataset_name,)
        ).fetchone()
        if not source:
            raise ValueError(f"unknown dataset: {dataset_name}")
        stats = self.statistics()
        return {"dataset": dict(source), "generated_at": utc_now(), "statistics": stats}
