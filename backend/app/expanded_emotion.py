"""Reproducible training for the prepared 5,500-meme expansion.

This module deliberately keeps MOOD's six binary labels separate from MemeMatch
scores.  The latter may be retained in the catalog for retrieval but are never
silently used as supervised targets here.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler

from app.emotion import EMOTIONS, SENTIMENTS, tune_thresholds

TEXT_DIMENSION = 384
IMAGE_DIMENSION = 768
FEATURE_VERSION = "dameme-5500-minilm-blipvision-v1"
MOOD_SOURCES = frozenset({"mood", "mood-tac24", "mood_tac24"})


def _key(value: str) -> str:
    return "".join(character.lower() for character in value if character.isalnum())


def _column(row: dict[str, str], *names: str) -> str | None:
    expected = {_key(name) for name in names}
    return next((value for name, value in row.items() if _key(name) in expected), None)


def _identifier(row: dict[str, str]) -> str:
    value = _column(row, "meme_id", "meme id", "id", "source_key")
    if not value or not value.strip():
        raise ValueError("every artifact row requires a non-empty meme_id")
    return value.strip()


def _include_in_training(row: dict[str, str]) -> bool:
    value = _column(row, "include_in_training")
    if value is None:
        raise ValueError("training manifest rows require include_in_training")
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no", ""}:
        return False
    raise ValueError(f"invalid include_in_training value: {value!r}")


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"required artifact is missing: {path}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"artifact has no rows: {path}")
    return rows


def _binary(value: str | None, label: str) -> int:
    if value is None:
        raise ValueError(f"MOOD row is missing {label} annotation")
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", label}:
        return 1
    if normalized in {"0", "false", "no", f"not_{label}", ""}:
        return 0
    raise ValueError(f"invalid MOOD {label} annotation: {value!r}")


@dataclass(frozen=True)
class ExpansionDataset:
    meme_ids: tuple[str, ...]
    splits: tuple[str, ...]
    sources: tuple[str, ...]
    text: np.ndarray
    image: np.ndarray
    labels: np.ndarray
    sentiment: np.ndarray
    sentiment_available: np.ndarray

    @property
    def mood_mask(self) -> np.ndarray:
        return np.asarray(
            [source.lower().replace("_", "-") in MOOD_SOURCES for source in self.sources]
        )


def load_expansion_artifacts(base: Path, *, strict_counts: bool = True) -> ExpansionDataset:
    """Load Drive artifacts and fail closed on alignment, leakage, or label errors."""
    inputs = _read_csv(base / "dameme_5500_model_inputs.csv")
    splits = _read_csv(base / "dameme_5500_dataset_splits.csv")
    manifest = _read_csv(base / "dameme_5500_training_manifest.csv")
    training_manifest = [row for row in manifest if _include_in_training(row)]
    with np.load(base / "dameme_5500_minilm_embeddings.npz") as archive:
        id_key = next(
            (name for name in archive.files if _key(name) in {"memeid", "memeids", "ids"}), None
        )
        embedding_key = next((name for name in archive.files if "embedding" in _key(name)), None)
        if id_key is None or embedding_key is None:
            raise ValueError("MiniLM NPZ must contain meme IDs and embeddings")
        text_ids = tuple(str(value) for value in archive[id_key].tolist())
        text = np.asarray(archive[embedding_key], dtype=np.float32)
    image = np.asarray(np.load(base / "dameme_5500_blip_embeddings.npy"), dtype=np.float32)
    input_ids = tuple(_identifier(row) for row in inputs)
    if len(set(input_ids)) != len(input_ids):
        raise ValueError("model inputs contain duplicate meme IDs")
    if input_ids != text_ids:
        raise ValueError("MiniLM meme IDs are not in exact model-input row order")
    if text.shape != (len(input_ids), TEXT_DIMENSION):
        raise ValueError(f"MiniLM embeddings must have shape (N, {TEXT_DIMENSION})")
    if image.shape != (len(input_ids), IMAGE_DIMENSION):
        raise ValueError(f"BLIP embeddings must have shape (N, {IMAGE_DIMENSION})")
    if not np.isfinite(text).all() or not np.isfinite(image).all():
        raise ValueError("feature artifacts contain missing or non-finite values")
    if np.any(np.linalg.norm(text, axis=1) < 1e-12) or np.any(
        np.linalg.norm(image, axis=1) < 1e-12
    ):
        raise ValueError("feature artifacts contain zero vectors")
    split_ids = tuple(_identifier(row) for row in splits)
    if len(set(split_ids)) != len(split_ids):
        raise ValueError("split file contains duplicate meme IDs")
    training_manifest_ids = tuple(_identifier(row) for row in training_manifest)
    if len(set(training_manifest_ids)) != len(training_manifest_ids):
        raise ValueError("training manifest candidates contain duplicate meme IDs")
    split_by_id = {
        meme_id: (_column(row, "split", "dataset_split") or "").lower()
        for meme_id, row in zip(split_ids, splits, strict=True)
    }
    manifest_by_id = {
        meme_id: row for meme_id, row in zip(training_manifest_ids, training_manifest, strict=True)
    }
    if set(input_ids) != set(split_by_id) or set(input_ids) != set(manifest_by_id):
        raise ValueError(
            "model inputs, split file, and included manifest candidates must contain the same meme IDs"
        )
    ordered_splits = tuple(split_by_id[item] for item in input_ids)
    if any(split not in {"train", "validation", "test"} for split in ordered_splits):
        raise ValueError("splits must be train, validation, or test")
    expected_splits = {"train": 4373, "validation": 552, "test": 544}
    actual_splits = {split: ordered_splits.count(split) for split in expected_splits}
    if strict_counts and actual_splits != expected_splits:
        raise ValueError(f"prepared 5,469-row split counts are required; found {actual_splits!r}")
    # A content or perceptual hash may occur only in one split.
    hashes: dict[str, str] = {}
    for item in input_ids:
        row = manifest_by_id[item]
        value = _column(row, "sha256", "perceptual_hash", "phash")
        if value:
            previous = hashes.setdefault(value, split_by_id[item])
            if previous != split_by_id[item]:
                raise ValueError("duplicate image hash leaks across dataset splits")
    sources, labels, sentiments, available = [], [], [], []
    for item in input_ids:
        row = manifest_by_id[item]
        source = (_column(row, "source", "dataset_source", "dataset") or "").strip()
        sources.append(source)
        if source.lower().replace("_", "-") in MOOD_SOURCES:
            labels.append([_binary(_column(row, label), label) for label in EMOTIONS])
            sentiment = (_column(row, "sentiment", "sentiment_label") or "").strip().lower()
            if sentiment and sentiment not in SENTIMENTS:
                raise ValueError(f"invalid MOOD sentiment label: {sentiment!r}")
            sentiments.append(SENTIMENTS.index(sentiment) if sentiment else -1)
            available.append(bool(sentiment))
        else:
            # Annotation systems differ: placeholders are never training targets.
            labels.append([0] * len(EMOTIONS))
            sentiments.append(-1)
            available.append(False)
    return ExpansionDataset(
        input_ids,
        ordered_splits,
        tuple(sources),
        text,
        image,
        np.asarray(labels, dtype=np.int8),
        np.asarray(sentiments),
        np.asarray(available),
    )


def _metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    thresholds: np.ndarray,
    sentiment: np.ndarray,
    sentiment_probabilities: np.ndarray | None,
) -> dict[str, Any]:
    predictions = probabilities >= thresholds
    per_emotion = {}
    for index, label in enumerate(EMOTIONS):
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels[:, index], predictions[:, index], average="binary", zero_division=0
        )
        per_emotion[label] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(labels[:, index].sum()),
        }
    result: dict[str, Any] = {
        "emotion_micro_f1": float(f1_score(labels, predictions, average="micro", zero_division=0)),
        "emotion_macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "per_emotion": per_emotion,
    }
    valid = sentiment >= 0
    if sentiment_probabilities is not None and valid.any():
        predicted = np.argmax(sentiment_probabilities[valid], axis=1)
        result.update(
            sentiment_accuracy=float(accuracy_score(sentiment[valid], predicted)),
            sentiment_macro_f1=float(
                f1_score(sentiment[valid], predicted, average="macro", zero_division=0)
            ),
            sentiment_samples=int(valid.sum()),
        )
    return result


def _features(data: ExpansionDataset, model: str) -> np.ndarray:
    if model == "text":
        return data.text
    if model == "vision":
        return data.image
    if model == "concat":
        return np.concatenate((data.text, data.image), axis=1)
    raise ValueError(f"unsupported model: {model}")


def train_baseline(
    data: ExpansionDataset, model: str, output: Path, seed: int = 42
) -> dict[str, Any]:
    mood = data.mood_mask
    train = mood & (np.asarray(data.splits) == "train")
    validation = mood & (np.asarray(data.splits) == "validation")
    test = mood & (np.asarray(data.splits) == "test")
    if not train.any() or not validation.any() or not test.any():
        raise ValueError("MOOD train, validation, and test rows are all required")
    x = _features(data, model)
    scaler = StandardScaler().fit(x[train])
    classifier = MultiOutputClassifier(
        LogisticRegression(max_iter=1000, random_state=seed, class_weight="balanced")
    ).fit(scaler.transform(x[train]), data.labels[train])
    probabilities = np.column_stack(
        [item.predict_proba(scaler.transform(x))[..., 1] for item in classifier.estimators_]
    )
    thresholds = tune_thresholds(probabilities[validation], data.labels[validation])
    sentiment_rows = train & data.sentiment_available
    sentiment_model = None
    sentiment_probabilities = None
    if len(np.unique(data.sentiment[sentiment_rows])) > 1:
        sentiment_model = LogisticRegression(
            max_iter=1000, random_state=seed, class_weight="balanced"
        ).fit(scaler.transform(x[sentiment_rows]), data.sentiment[sentiment_rows])
        sentiment_probabilities = sentiment_model.predict_proba(scaler.transform(x[test]))
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"dameme_5500_{model}_seed{seed}.pkl"
    payload = {
        "format": 1,
        "model": model,
        "feature_version": FEATURE_VERSION,
        "label_order": list(EMOTIONS),
        "thresholds": thresholds.tolist(),
        "scaler": scaler,
        "emotion": classifier,
        "sentiment": sentiment_model,
    }
    path.write_bytes(pickle.dumps(payload))
    return {
        "model": model,
        "artifact": str(path),
        "thresholds": dict(zip(EMOTIONS, thresholds.tolist(), strict=True)),
        **_metrics(
            data.labels[test],
            probabilities[test],
            thresholds,
            data.sentiment[test],
            sentiment_probabilities,
        ),
    }


def predict_text_embedding(artifact: Path, embedding: np.ndarray) -> dict[str, object]:
    """Predict an emotion distribution from a trusted exported text-only model."""
    if not artifact.is_file():
        raise FileNotFoundError(f"text emotion artifact is missing: {artifact}")
    payload = pickle.loads(artifact.read_bytes())
    if payload.get("model") != "text" or payload.get("label_order") != list(EMOTIONS):
        raise ValueError("artifact is not a compatible DaMeme 5,500 text model")
    values = np.asarray(embedding, dtype=np.float32)
    if values.shape != (TEXT_DIMENSION,) or not np.isfinite(values).all():
        raise ValueError("text embedding must be a finite 384-dimensional vector")
    classifier = payload.get("emotion")
    if classifier is None or payload.get("scaler") is None:
        raise ValueError("text emotion artifact is incomplete")
    transformed = payload["scaler"].transform(values[None])
    probabilities = np.asarray(
        [item.predict_proba(transformed)[0, 1] for item in classifier.estimators_], dtype=np.float32
    )
    thresholds = np.asarray(payload.get("thresholds"), dtype=np.float32)
    if thresholds.shape != (len(EMOTIONS),):
        raise ValueError("text emotion artifact has invalid thresholds")
    values_by_label = dict(zip(EMOTIONS, probabilities.astype(float).tolist(), strict=True))
    return {
        "model": "text",
        "feature_version": payload["feature_version"],
        "emotion_probabilities": values_by_label,
        "emotion_labels": [
            label
            for label, value, threshold in zip(EMOTIONS, probabilities, thresholds, strict=True)
            if value >= threshold
        ],
        "thresholds": dict(zip(EMOTIONS, thresholds.astype(float).tolist(), strict=True)),
    }


def train_gated(
    data: ExpansionDataset,
    output: Path,
    seed: int = 42,
    epochs: int = 25,
    resume: Path | None = None,
) -> dict[str, Any]:
    """Train the fourth comparison model, retaining a resumable CPU/GPU checkpoint."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from app.emotion_fusion import GatedEmotionFusion

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    mood = data.mood_mask
    splits = np.asarray(data.splits)
    train, validation, test = (
        mood & (splits == "train"),
        mood & (splits == "validation"),
        mood & (splits == "test"),
    )
    if not train.any() or not validation.any() or not test.any():
        raise ValueError("MOOD train, validation, and test rows are all required")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GatedEmotionFusion(TEXT_DIMENSION, IMAGE_DIMENSION).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    positives = data.labels[train].sum(axis=0)
    weights = np.divide(
        train.sum() - positives, positives, out=np.ones(6, dtype=np.float32), where=positives > 0
    )
    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(weights, device=device, dtype=torch.float32)
    )
    start, best_loss, best_state, history = (
        0,
        float("inf"),
        None,
        {"train_loss": [], "validation_loss": []},
    )
    if resume:
        checkpoint = torch.load(resume, map_location=device, weights_only=True)
        if checkpoint.get("feature_version") != FEATURE_VERSION:
            raise ValueError("checkpoint uses incompatible feature version")
        model.load_state_dict(checkpoint["state"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start = int(checkpoint["epoch"]) + 1
        best_loss = float(checkpoint["best_validation_loss"])
        best_state = checkpoint.get("best_state")
        history = checkpoint.get("history", history)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(data.text[train]),
            torch.from_numpy(data.image[train]),
            torch.from_numpy(data.labels[train].astype(np.float32)),
        ),
        batch_size=64,
        shuffle=True,
    )
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / f"dameme_5500_gated_seed{seed}.pt"
    for epoch in range(start, start + epochs):
        model.train()
        losses = []
        for text_batch, image_batch, labels_batch in loader:
            optimizer.zero_grad()
            logits, _ = model(text_batch.to(device), image_batch.to(device))
            loss = loss_fn(logits, labels_batch.to(device))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.inference_mode():
            validation_logits, _ = model(
                torch.from_numpy(data.text[validation]).to(device),
                torch.from_numpy(data.image[validation]).to(device),
            )
            validation_loss = float(
                loss_fn(
                    validation_logits,
                    torch.from_numpy(data.labels[validation].astype(np.float32)).to(device),
                ).cpu()
            )
        history["train_loss"].append(float(np.mean(losses)))
        history["validation_loss"].append(validation_loss)
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
        torch.save(
            {
                "format": 1,
                "model": "gated",
                "feature_version": FEATURE_VERSION,
                "label_order": list(EMOTIONS),
                "state": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_state": best_state,
                "epoch": epoch,
                "best_validation_loss": best_loss,
                "history": history,
            },
            checkpoint_path,
        )
    if best_state is None:
        raise RuntimeError("gated training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        validation_logits, _ = model(
            torch.from_numpy(data.text[validation]).to(device),
            torch.from_numpy(data.image[validation]).to(device),
        )
        test_logits, _ = model(
            torch.from_numpy(data.text[test]).to(device),
            torch.from_numpy(data.image[test]).to(device),
        )
    validation_probabilities = torch.sigmoid(validation_logits).cpu().numpy()
    test_probabilities = torch.sigmoid(test_logits).cpu().numpy()
    thresholds = tune_thresholds(validation_probabilities, data.labels[validation])
    final = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    final.update(
        state=best_state,
        thresholds=thresholds.tolist(),
        metrics=_metrics(
            data.labels[test], test_probabilities, thresholds, data.sentiment[test], None
        ),
    )
    torch.save(final, checkpoint_path)
    return {
        "model": "gated",
        "artifact": str(checkpoint_path),
        "device": str(device),
        "thresholds": dict(zip(EMOTIONS, thresholds.tolist(), strict=True)),
        "loss_curves": history,
        **final["metrics"],
    }


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train DaMeme's prepared 5,500-catalog emotion baselines"
    )
    parser.add_argument(
        "--base", type=Path, required=True, help="Google Drive expansion_5000 directory"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=("text", "vision", "concat", "gated"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    data = load_expansion_artifacts(args.base)
    result = (
        train_gated(data, args.output, args.seed, args.epochs, args.resume)
        if args.model == "gated"
        else train_baseline(data, args.model, args.output, args.seed)
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
