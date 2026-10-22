import argparse
import json
from pathlib import Path

from app.multimodal import AnalysisService, BlipCaptionProvider, EasyOcrProvider, analyze_image


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DaMeme multimodal analysis")
    sub = parser.add_subparsers(dest="command", required=True)
    image = sub.add_parser("analyze-image")
    image.add_argument("--image", type=Path, required=True)
    image.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    batch = sub.add_parser("analyze-batch")
    batch.add_argument("--database", type=Path, required=True)
    batch.add_argument("--limit", type=int, default=10)
    batch.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    batch.add_argument("--force", action="store_true")
    stats = sub.add_parser("analysis-stats")
    stats.add_argument("--database", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "analyze-image":
        print(
            json.dumps(
                analyze_image(
                    args.image, EasyOcrProvider(args.device), BlipCaptionProvider(args.device)
                ),
                indent=2,
            )
        )
        return 0
    device = getattr(args, "device", "cpu")
    service = AnalysisService(args.database, EasyOcrProvider(device), BlipCaptionProvider(device))
    try:
        if args.command == "analyze-batch":
            print(json.dumps(service.analyze_batch(args.limit, args.force), indent=2))
        else:
            rows = service.connection.execute(
                "SELECT status, COUNT(*) AS count FROM multimodal_analyses GROUP BY status"
            )
            print(json.dumps([dict(row) for row in rows], indent=2))
    finally:
        service.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
