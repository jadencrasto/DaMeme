import argparse
import json
import sys
from pathlib import Path

from app.datasets.catalog_generations import CatalogPlanner, PermissionReview
from app.datasets.database import connect, migrate
from app.datasets.ingestion import DatasetIngester, IngestionOptions, SourceDetails
from app.datasets.validation import ImageValidationError, validate_image


def _database(value: str) -> Path:
    return Path(value).resolve()


def _load_metadata(path: str | None) -> dict[str, dict[str, object]]:
    if not path:
        return {}
    content = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(content, dict):
        raise ValueError("metadata JSON must map source IDs to annotation objects")
    return content


def _load_permission_review(path: str) -> PermissionReview:
    content = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(content, dict):
        raise ValueError("permission review JSON must be an object")
    try:
        return PermissionReview(**content)
    except TypeError as exc:
        raise ValueError("permission review JSON has invalid fields") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dameme-datasets", description="Safe local DaMeme dataset management"
    )
    parser.add_argument(
        "--database", default="dameme.db", type=_database, help="SQLite database path"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init-db", help="apply local SQLite schema migrations")
    validate = commands.add_parser("validate", help="validate files under a local directory")
    validate.add_argument("input", type=Path)
    validate.add_argument("--max-bytes", type=int, default=20 * 1024 * 1024)
    validate.add_argument("--max-pixels", type=int, default=40_000_000)
    ingest = commands.add_parser("import", help="import a local image directory (no downloads)")
    ingest.add_argument("input", type=Path)
    ingest.add_argument("--dataset", required=True)
    ingest.add_argument("--version", required=True)
    ingest.add_argument("--source-url", required=True)
    ingest.add_argument(
        "--license-status", required=True, choices=["verified", "restricted", "unknown", "pending"]
    )
    ingest.add_argument("--license-notes", default="")
    ingest.add_argument("--access-notes", default="")
    ingest.add_argument("--manifest-reference")
    ingest.add_argument("--source-reported-count", type=int)
    ingest.add_argument(
        "--metadata", help="JSON mapping relative file paths to supplied annotations"
    )
    ingest.add_argument("--batch-size", type=int, default=100)
    ingest.add_argument(
        "--max-images", type=int, default=100, help="safety cap; raise only with authorization"
    )
    ingest.add_argument("--near-distance", type=int, default=6)
    ingest.add_argument("--max-bytes", type=int, default=20 * 1024 * 1024)
    ingest.add_argument("--max-pixels", type=int, default=40_000_000)
    dry_run = commands.add_parser(
        "catalog-dry-run", help="read-only validation and duplicate plan; writes no database or catalog"
    )
    dry_run.add_argument("input", type=Path)
    dry_run.add_argument("--permission-review", required=True)
    dry_run.add_argument("--near-distance", type=int, default=6)
    dry_run.add_argument("--max-bytes", type=int, default=20 * 1024 * 1024)
    dry_run.add_argument("--max-pixels", type=int, default=40_000_000)
    stats = commands.add_parser("stats", help="show aggregate statistics")
    stats.add_argument("--json", action="store_true")
    manifest = commands.add_parser("export-manifest", help="write a JSON provenance summary")
    manifest.add_argument("--dataset", required=True)
    manifest.add_argument("--output", type=Path, required=True)
    failures = commands.add_parser("failures", help="show validation/import failures")
    failures.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "init-db":
        connection = connect(args.database)
        migrate(connection)
        connection.close()
        print(f"Initialized {args.database}")
        return 0
    if args.command == "validate":
        valid, invalid = 0, []
        for path in sorted(candidate for candidate in args.input.rglob("*") if candidate.is_file()):
            try:
                validate_image(path, max_bytes=args.max_bytes, max_pixels=args.max_pixels)
                valid += 1
            except ImageValidationError as exc:
                invalid.append({"path": str(path), "reason": str(exc)})
        print(json.dumps({"valid": valid, "invalid": invalid}, indent=2))
        return 1 if invalid else 0
    if args.command == "catalog-dry-run":
        result = CatalogPlanner().dry_run(
            args.input,
            _load_permission_review(args.permission_review),
            near_duplicate_distance=args.near_distance,
            max_bytes=args.max_bytes,
            max_pixels=args.max_pixels,
        )
        print(
            json.dumps(
                {
                    "accepted": [item.__dict__ for item in result.accepted],
                    "rejected": [item.__dict__ for item in result.rejected],
                    "exact_duplicates": result.exact_duplicates,
                    "near_duplicates": result.near_duplicates,
                },
                indent=2,
            )
        )
        return 1 if result.rejected else 0
    ingester = DatasetIngester(args.database)
    try:
        if args.command == "import":
            result = ingester.ingest(
                args.input,
                SourceDetails(
                    args.dataset,
                    args.version,
                    args.source_url,
                    args.license_status,
                    args.license_notes,
                    args.access_notes,
                    args.manifest_reference,
                    args.source_reported_count,
                ),
                IngestionOptions(
                    args.batch_size,
                    args.max_images,
                    args.max_bytes,
                    args.max_pixels,
                    args.near_distance,
                ),
                _load_metadata(args.metadata),
            )
            print(json.dumps(result.__dict__, indent=2))
            return 0
        if args.command == "stats":
            payload = ingester.statistics()
            print(json.dumps(payload, indent=2) if args.json else _human_stats(payload))
            return 0
        if args.command == "export-manifest":
            payload = ingester.export_manifest(args.dataset)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"Wrote {args.output}")
            return 0
        if args.command == "failures":
            rows = [
                dict(row)
                for row in ingester.connection.execute(
                    "SELECT * FROM ingestion_failures ORDER BY id"
                )
            ]
            print(
                json.dumps(rows, indent=2)
                if args.json
                else "\n".join(f"{row['input_path']}: {row['reason']}" for row in rows)
            )
            return 0
    finally:
        ingester.close()
    return 2


def _human_stats(payload: dict[str, object]) -> str:
    return "\n".join(
        f"{key.replace('_', ' ')}: {value}"
        for key, value in payload.items()
        if not isinstance(value, list)
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
