import argparse
import json
from pathlib import Path

from app.emotion import EmotionService


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DaMeme Phase 4 emotion pipeline")
    parser.add_argument("--database", type=Path, default=Path("dameme.db"))
    parser.add_argument("--artifacts", type=Path, default=Path("../artifacts/emotion"))
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "extract-features"):
        item = commands.add_parser(name)
        item.add_argument("--limit", type=int, default=50)
        item.add_argument("--seed", type=int, default=42)
        if name == "extract-features":
            item.add_argument("--feature-version", default="phase4-hash-text-rgb-v1")
            item.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    train = commands.add_parser("train")
    train.add_argument(
        "--model",
        choices=[
            "frequency-baseline",
            "text-baseline",
            "image-baseline",
            "simple-fusion",
            "emotion-fusion",
        ],
        required=True,
    )
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--feature-version", default="phase4-hash-text-rgb-v1")
    train.add_argument("--epochs", type=int, default=8)
    train.add_argument("--resume-from", type=Path)
    train.add_argument("--emotion-pos-weights")
    train.add_argument("--emotion-loss-weight", type=float, default=1.0)
    train.add_argument("--sentiment-loss-weight", type=float, default=1.0)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--model", required=True)
    evaluate.add_argument("--feature-version", default="phase4-minilm-blipvision-v1")
    evaluate.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    service = EmotionService(args.database)
    try:
        if args.command == "prepare":
            result = service.prepare(args.limit, args.seed)
        elif args.command == "extract-features":
            result = service.extract_features(args.limit, args.feature_version, args.device)
        elif args.command == "train":
            result = (
                service.train_gated(
                    args.artifacts,
                    args.seed,
                    args.epochs,
                    args.feature_version,
                    args.resume_from,
                    [float(value) for value in args.emotion_pos_weights.split(",")]
                    if args.emotion_pos_weights
                    else None,
                    args.emotion_loss_weight,
                    args.sentiment_loss_weight,
                )
                if args.model == "emotion-fusion"
                else service.train(args.model, args.artifacts, args.seed, args.feature_version)
            )
        else:
            if args.model == "emotion-fusion":
                result = service.evaluate_gated(args.feature_version)
            else:
                model = service.connection.execute(
                    "SELECT artifact_path FROM emotion_models WHERE model_name=? AND feature_version=? ORDER BY id DESC LIMIT 1",
                    (args.model, args.feature_version),
                ).fetchone()
                if not model:
                    raise FileNotFoundError(f"no artifact for {args.model}")
                result = service.evaluate_artifact(
                    Path(model["artifact_path"]), service.rows(args.feature_version)
                )
            if args.output_dir:
                args.output_dir.mkdir(parents=True, exist_ok=True)
                stem = f"{args.model}-heldout-report"
                (args.output_dir / f"{stem}.json").write_text(json.dumps(result, indent=2))
                markdown = "\n".join(
                    [
                        f"# {args.model} held-out report",
                        "",
                        f"- Samples: {result.get('test_samples', result.get('samples', 0))}",
                        f"- Emotion micro F1: {result.get('emotion_micro_f1', 'unavailable')}",
                        f"- Emotion macro F1: {result.get('emotion_macro_f1', 'unavailable')}",
                        f"- Sentiment accuracy: {result.get('sentiment_accuracy', 'unavailable')}",
                        f"- Sentiment macro F1: {result.get('sentiment_macro_f1', 'unavailable')}",
                        "",
                        str(result.get("warning", "")),
                    ]
                )
                (args.output_dir / f"{stem}.md").write_text(markdown + "\n")
            print(json.dumps(result, indent=2))
            return 0
        print(json.dumps(result, indent=2))
    finally:
        service.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
