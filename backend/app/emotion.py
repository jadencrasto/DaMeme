"""Small, reproducible Phase 4 feature, split, baseline, and prediction utilities."""

import hashlib
import json
import pickle
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    hamming_loss,
    precision_recall_fscore_support,
)
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler

from app.datasets.database import connect, migrate

EMOTIONS = ("anger", "fear", "joy", "sadness", "surprise", "disgust")
SENTIMENTS = ("positive", "negative", "neutral")
FEATURE_VERSION = "phase4-hash-text-rgb-v1"
PRETRAINED_FEATURE_VERSION = "phase4-minilm-blipvision-v1"


def now() -> str:
    return datetime.now(UTC).isoformat()


def mood_targets(annotation_json: str) -> tuple[np.ndarray, int]:
    annotation = json.loads(annotation_json)
    values = annotation["emotions"]
    emotions = np.array([int(values[label] == label) for label in EMOTIONS], dtype=np.int8)
    return emotions, SENTIMENTS.index(annotation["sentiment"])


def deterministic_splits(rows: list[dict[str, object]], seed: int = 42) -> dict[int, str]:
    """Hash-group records so equal SHA/pHash identifiers cannot cross a split."""
    groups: dict[str, list[int]] = {}
    for row in rows:
        group = str(row.get("sha256") or row.get("perceptual_hash") or row["id"])
        groups.setdefault(group, []).append(int(row["id"]))
    result: dict[int, str] = {}
    for group, identifiers in groups.items():
        bucket = int(hashlib.sha256(f"{seed}:{group}".encode()).hexdigest(), 16) % 10
        split = "test" if bucket == 0 else "validation" if bucket == 1 else "train"
        result.update(dict.fromkeys(identifiers, split))
    return result


def text_features(text: str, dimensions: int = 128) -> np.ndarray:
    vectorizer = HashingVectorizer(n_features=dimensions, alternate_sign=False, norm="l2")
    return vectorizer.transform([text or ""]).toarray()[0].astype(np.float32)


def image_features(path: Path, size: int = 8) -> np.ndarray:
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("RGB").resize((size, size)), dtype=np.float32) / 255.0
    return pixels.reshape(-1)


class PretrainedFeatureExtractor:
    """Frozen MiniLM and existing BLIP vision feature extractor, loaded lazily."""

    text_model_id = "sentence-transformers/all-MiniLM-L6-v2"
    image_model_id = "Salesforce/blip-image-captioning-base"

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self._text_tokenizer = self._text_model = self._processor = self._blip = None

    def text(self, content: str) -> np.ndarray:
        import torch
        from transformers import AutoModel, AutoTokenizer

        if self._text_model is None:
            self._text_tokenizer = AutoTokenizer.from_pretrained(self.text_model_id)
            self._text_model = AutoModel.from_pretrained(self.text_model_id).to(self.device).eval()
        encoded = self._text_tokenizer(
            content or "", return_tensors="pt", truncation=True, max_length=256
        ).to(self.device)
        with torch.inference_mode():
            output = self._text_model(**encoded).last_hidden_state
            weights = encoded["attention_mask"].unsqueeze(-1)
            pooled = (output * weights).sum(1) / weights.sum(1).clamp(min=1)
        vector = pooled[0].cpu().numpy().astype(np.float32)
        return vector / max(np.linalg.norm(vector), 1e-12)

    def image(self, path: Path) -> np.ndarray:
        import torch
        from transformers import BlipForConditionalGeneration, BlipProcessor

        if self._blip is None:
            self._processor = BlipProcessor.from_pretrained(self.image_model_id)
            self._blip = (
                BlipForConditionalGeneration.from_pretrained(self.image_model_id)
                .to(self.device)
                .eval()
            )
        with Image.open(path) as source:
            values = self._processor(images=source.convert("RGB"), return_tensors="pt")[
                "pixel_values"
            ].to(self.device)
        with torch.inference_mode():
            hidden = self._blip.vision_model(pixel_values=values).last_hidden_state.mean(dim=1)
        vector = hidden[0].cpu().numpy().astype(np.float32)
        return vector / max(np.linalg.norm(vector), 1e-12)


def feature_hash(text: np.ndarray, image: np.ndarray) -> str:
    return hashlib.sha256(text.tobytes() + image.tobytes()).hexdigest()


def tune_thresholds(probabilities: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Select each sigmoid threshold using validation labels only."""
    values = []
    for index in range(labels.shape[1]):
        scores = [
            f1_score(labels[:, index], probabilities[:, index] >= value, zero_division=0)
            for value in (0.3, 0.4, 0.5, 0.6, 0.7)
        ]
        values.append((0.3, 0.4, 0.5, 0.6, 0.7)[int(np.argmax(scores))])
    return np.array(values, dtype=np.float32)


@dataclass(frozen=True)
class FeatureRow:
    meme_id: int
    text: np.ndarray
    image: np.ndarray
    emotions: np.ndarray
    sentiment: int
    split: str


class EmotionService:
    def __init__(self, database: Path):
        self.connection = connect(database)
        migrate(self.connection)

    def close(self) -> None:
        self.connection.close()

    def prepare(self, limit: int, seed: int) -> dict[str, int]:
        rows = [
            dict(row)
            for row in self.connection.execute(
                "SELECT * FROM memes WHERE emotion_label IS NOT NULL ORDER BY id LIMIT ?", (limit,)
            )
        ]
        splits = deterministic_splits(rows, seed)
        with self.connection:
            for meme_id, split in splits.items():
                self.connection.execute(
                    "UPDATE memes SET dataset_split=? WHERE id=?", (split, meme_id)
                )
        return {
            split: list(splits.values()).count(split) for split in ("train", "validation", "test")
        }

    def extract_features(
        self, limit: int, feature_version: str = FEATURE_VERSION, device: str = "cpu"
    ) -> dict[str, int | str]:
        rows = self.connection.execute(
            "SELECT m.*, a.ocr_text, a.visual_caption FROM memes m JOIN multimodal_analyses a ON a.meme_id=m.id WHERE m.emotion_label IS NOT NULL AND a.status='completed' ORDER BY m.id LIMIT ?",
            (limit,),
        ).fetchall()
        completed = skipped = 0
        extractor = (
            PretrainedFeatureExtractor(device)
            if feature_version == PRETRAINED_FEATURE_VERSION
            else None
        )
        for row in rows:
            existing = self.connection.execute(
                "SELECT 1 FROM emotion_features WHERE meme_id=? AND feature_version=?",
                (row["id"], feature_version),
            ).fetchone()
            if existing:
                skipped += 1
                continue
            content = " ".join(filter(None, (row["ocr_text"], row["visual_caption"])))
            text = extractor.text(content) if extractor else text_features(content)
            image = (
                extractor.image(Path(row["local_image_path"]))
                if extractor
                else image_features(Path(row["local_image_path"]))
            )
            digest = feature_hash(text, image)
            with self.connection:
                self.connection.execute(
                    "INSERT INTO emotion_features(meme_id,feature_version,text_encoder,image_encoder,text_json,image_json,feature_hash,status,extracted_at) VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        row["id"],
                        feature_version,
                        extractor.text_model_id if extractor else "HashingVectorizer-128",
                        extractor.image_model_id if extractor else "RGB-8x8",
                        json.dumps(text.tolist()),
                        json.dumps(image.tolist()),
                        digest,
                        "completed",
                        now(),
                    ),
                )
            completed += 1
        return {"completed": completed, "skipped": skipped, "feature_version": feature_version}

    def rows(self, feature_version: str = FEATURE_VERSION) -> list[FeatureRow]:
        records = self.connection.execute(
            "SELECT m.id,m.annotation_json,m.dataset_split,f.text_json,f.image_json FROM memes m JOIN emotion_features f ON f.meme_id=m.id WHERE f.feature_version=?",
            (feature_version,),
        ).fetchall()
        return [
            FeatureRow(
                row["id"],
                np.array(json.loads(row["text_json"]), dtype=np.float32),
                np.array(json.loads(row["image_json"]), dtype=np.float32),
                *mood_targets(row["annotation_json"]),
                row["dataset_split"] or "train",
            )
            for row in records
        ]

    @staticmethod
    def _matrix(rows: list[FeatureRow], model_name: str) -> np.ndarray:
        return np.array(
            [
                row.text
                if model_name == "text-baseline"
                else row.image
                if model_name == "image-baseline"
                else np.concatenate((row.text, row.image))
                for row in rows
            ]
        )

    def train(
        self,
        model_name: str,
        artifacts: Path,
        seed: int = 42,
        feature_version: str = FEATURE_VERSION,
    ) -> dict[str, object]:
        rows = self.rows(feature_version)
        if not rows:
            raise RuntimeError(f"No cached features are available for {feature_version}")
        train = [row for row in rows if row.split == "train"] or rows
        if model_name == "frequency-baseline":
            emotion_rates = np.mean([row.emotions for row in train], axis=0)
            sentiment_counts = np.bincount(
                [row.sentiment for row in train], minlength=len(SENTIMENTS)
            )
            payload = {
                "model_name": model_name,
                "emotion_rates": emotion_rates,
                "sentiment_label": int(np.argmax(sentiment_counts)),
            }
            version = f"{model_name}-seed{seed}-n{len(train)}-v1"
            artifacts.mkdir(parents=True, exist_ok=True)
            path = artifacts / f"{version}.pkl"
            path.write_bytes(pickle.dumps(payload))
            metrics = self.evaluate_artifact(path, rows)
            with self.connection:
                self.connection.execute(
                    "INSERT OR REPLACE INTO emotion_models(model_name,model_version,artifact_path,feature_version,configuration_json,metrics_json,trained_at) VALUES (?,?,?,?,?,?,?)",
                    (
                        model_name,
                        version,
                        str(path),
                        feature_version,
                        json.dumps({"seed": seed, "feature_version": feature_version}),
                        json.dumps(metrics),
                        now(),
                    ),
                )
            return {"model_version": version, **metrics}
        x, emotions, sentiments = (
            self._matrix(train, model_name),
            np.array([row.emotions for row in train]),
            np.array([row.sentiment for row in train]),
        )
        scaler = StandardScaler().fit(x)
        x = scaler.transform(x)
        emotion_model = MultiOutputClassifier(
            LogisticRegression(max_iter=300, random_state=seed)
        ).fit(x, emotions)
        sentiment_model = LogisticRegression(max_iter=300, random_state=seed).fit(x, sentiments)
        version = f"{model_name}-seed{seed}-n{len(train)}-v1"
        artifacts.mkdir(parents=True, exist_ok=True)
        path = artifacts / f"{version}.pkl"
        path.write_bytes(
            pickle.dumps(
                {
                    "scaler": scaler,
                    "emotion": emotion_model,
                    "sentiment": sentiment_model,
                    "model_name": model_name,
                }
            )
        )
        metrics = self.evaluate_artifact(path, rows)
        with self.connection:
            self.connection.execute(
                "INSERT OR REPLACE INTO emotion_models(model_name,model_version,artifact_path,feature_version,configuration_json,metrics_json,trained_at) VALUES (?,?,?,?,?,?,?)",
                (
                    model_name,
                    version,
                    str(path),
                    feature_version,
                    json.dumps({"seed": seed, "feature_version": feature_version}),
                    json.dumps(metrics),
                    now(),
                ),
            )
        return {"model_version": version, **metrics}

    def train_gated(
        self,
        artifacts: Path,
        seed: int = 42,
        epochs: int = 8,
        feature_version: str = PRETRAINED_FEATURE_VERSION,
        resume_from: Path | None = None,
        emotion_pos_weights: list[float] | None = None,
        emotion_loss_weight: float = 1.0,
        sentiment_loss_weight: float = 1.0,
    ) -> dict[str, object]:
        import random

        import torch

        from app.emotion_fusion import GatedEmotionFusion

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        rows = self.rows(feature_version)
        train = [row for row in rows if row.split == "train"] or rows
        validation = [row for row in rows if row.split == "validation"] or train
        model = GatedEmotionFusion(len(train[0].text), len(train[0].image), hidden_dim=64)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        positives = np.sum([row.emotions for row in train], axis=0)
        inferred_weights = np.divide(
            len(train) - positives,
            positives,
            out=np.ones_like(positives, dtype=np.float32),
            where=positives > 0,
        ).astype(np.float32)
        weights = np.asarray(emotion_pos_weights or inferred_weights, dtype=np.float32)
        if weights.shape != (len(EMOTIONS),) or np.any(weights <= 0):
            raise ValueError("emotion positive-class weights must contain six positive values")
        emotion_loss = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(weights, dtype=torch.float32)
        )
        sentiment_loss = torch.nn.CrossEntropyLoss()
        best_state, best_loss, patience, start_epoch = None, float("inf"), 0, 0
        if resume_from:
            checkpoint = torch.load(resume_from, map_location="cpu", weights_only=True)
            if checkpoint["text_dim"] != len(train[0].text) or checkpoint["image_dim"] != len(
                train[0].image
            ):
                raise ValueError("checkpoint feature dimensions do not match cached features")
            model.load_state_dict(checkpoint["state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            best_state = checkpoint.get("best_state", checkpoint["state"])
            best_loss = float(checkpoint.get("best_validation_loss", float("inf")))
            patience = int(checkpoint.get("patience", 0))
            start_epoch = int(checkpoint.get("epoch", -1)) + 1
        completed_epoch = start_epoch - 1
        for epoch in range(start_epoch, start_epoch + epochs):
            completed_epoch = epoch
            model.train()
            for row in train:
                optimizer.zero_grad()
                emo, sent = model(torch.tensor(row.text)[None], torch.tensor(row.image)[None])
                loss = emotion_loss_weight * emotion_loss(
                    emo, torch.tensor(row.emotions, dtype=torch.float32)[None]
                ) + sentiment_loss_weight * sentiment_loss(sent, torch.tensor([row.sentiment]))
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.inference_mode():
                value = sum(
                    float(
                        emotion_loss(
                            model(torch.tensor(row.text)[None], torch.tensor(row.image)[None])[0],
                            torch.tensor(row.emotions, dtype=torch.float32)[None],
                        )
                    )
                    for row in validation
                ) / len(validation)
            if value < best_loss:
                best_loss, patience = value, 0
                best_state = {key: value.cpu() for key, value in model.state_dict().items()}
            else:
                patience += 1
                if patience >= 3:
                    break
        model.load_state_dict(best_state)
        artifacts.mkdir(parents=True, exist_ok=True)
        path = artifacts / f"emotion-fusion-seed{seed}-n{len(train)}-v1.pt"
        thresholds = np.full(6, 0.5, dtype=np.float32)
        torch.save(
            {
                "state": model.state_dict(),
                "text_dim": len(train[0].text),
                "image_dim": len(train[0].image),
                "feature_version": feature_version,
                "thresholds": thresholds.tolist(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": completed_epoch,
                "best_validation_loss": best_loss,
                "best_state": best_state,
                "patience": patience,
                "training_configuration": {
                    "seed": seed,
                    "emotion_pos_weights": weights.tolist(),
                    "emotion_loss_weight": emotion_loss_weight,
                    "sentiment_loss_weight": sentiment_loss_weight,
                },
            },
            path,
        )
        with self.connection:
            self.connection.execute(
                "INSERT OR REPLACE INTO emotion_models(model_name,model_version,artifact_path,feature_version,configuration_json,metrics_json,trained_at) VALUES (?,?,?,?,?,?,?)",
                (
                    "emotion-fusion",
                    path.stem,
                    str(path),
                    feature_version,
                    json.dumps({"seed": seed, "epochs": epochs, "resumed": bool(resume_from)}),
                    json.dumps({"validation_emotion_loss": best_loss}),
                    now(),
                ),
            )
        return {
            "model_version": path.stem,
            "train_samples": len(train),
            "validation_samples": len(validation),
            "validation_emotion_loss": best_loss,
            "epochs_requested": epochs,
            "epochs_completed_through": completed_epoch,
            "resumed": bool(resume_from),
            "emotion_pos_weights": weights.tolist(),
            "checkpoint": str(path),
        }

    def predict(self, meme_id: int) -> dict[str, object]:
        import torch

        from app.emotion_fusion import GatedEmotionFusion

        model_row = self.connection.execute(
            "SELECT * FROM emotion_models WHERE model_name='emotion-fusion' ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if not model_row:
            raise RuntimeError("No trained emotion-fusion model is available")
        feature = self.connection.execute(
            "SELECT * FROM emotion_features WHERE meme_id=? AND feature_version=?",
            (meme_id, model_row["feature_version"]),
        ).fetchone()
        if not feature:
            meme = self.connection.execute("SELECT 1 FROM memes WHERE id=?", (meme_id,)).fetchone()
            if not meme:
                raise KeyError("meme not found")
            raise RuntimeError("Required cached feature is unavailable")
        cached = self.connection.execute(
            "SELECT * FROM emotion_predictions WHERE meme_id=? AND model_id=? AND feature_hash=?",
            (meme_id, model_row["id"], feature["feature_hash"]),
        ).fetchone()
        if cached:
            emotions = json.loads(cached["emotion_probabilities_json"])
            sentiment = json.loads(cached["sentiment_probabilities_json"])
            return {
                "id": cached["id"],
                "status": cached["status"],
                "emotion_probabilities": emotions,
                "emotion_labels": sentiment.get("emotion_labels", []),
                "sentiment_probabilities": sentiment.get("probabilities", {}),
                "sentiment_label": sentiment.get("label"),
                "predicted_at": cached["predicted_at"],
                "cached": True,
                "model_version": model_row["model_version"],
                "feature_version": model_row["feature_version"],
            }
        payload = torch.load(model_row["artifact_path"], map_location="cpu", weights_only=True)
        model = GatedEmotionFusion(payload["text_dim"], payload["image_dim"], hidden_dim=64)
        model.load_state_dict(payload["state"])
        model.eval()
        with torch.inference_mode():
            emotion_logits, sentiment_logits = model(
                torch.tensor(json.loads(feature["text_json"]), dtype=torch.float32)[None],
                torch.tensor(json.loads(feature["image_json"]), dtype=torch.float32)[None],
            )
        emotion_values = torch.sigmoid(emotion_logits)[0].tolist()
        sentiment_values = torch.softmax(sentiment_logits, dim=-1)[0].tolist()
        thresholds = payload.get("thresholds", [0.5] * 6)
        emotions = dict(zip(EMOTIONS, emotion_values, strict=True))
        sentiments = dict(zip(SENTIMENTS, sentiment_values, strict=True))
        result = {
            "status": "completed",
            "emotion_probabilities": emotions,
            "emotion_labels": [
                label
                for label, value, threshold in zip(
                    EMOTIONS, emotion_values, thresholds, strict=True
                )
                if value >= threshold
            ],
            "sentiment_probabilities": sentiments,
            "sentiment_label": SENTIMENTS[int(np.argmax(sentiment_values))],
            "model_version": model_row["model_version"],
            "feature_version": model_row["feature_version"],
            "cached": False,
        }
        with self.connection:
            cursor = self.connection.execute(
                "INSERT INTO emotion_predictions(meme_id,model_id,feature_hash,status,emotion_probabilities_json,sentiment_probabilities_json,predicted_at) VALUES (?,?,?,?,?,?,?)",
                (
                    meme_id,
                    model_row["id"],
                    feature["feature_hash"],
                    "completed",
                    json.dumps(emotions),
                    json.dumps(
                        {
                            "probabilities": sentiments,
                            "label": result["sentiment_label"],
                            "emotion_labels": result["emotion_labels"],
                        }
                    ),
                    now(),
                ),
            )
        result.update({"id": cursor.lastrowid, "predicted_at": now()})
        return result

    @staticmethod
    def _metric_report(
        actual_emotions: np.ndarray,
        predicted_emotions: np.ndarray,
        actual_sentiment: np.ndarray,
        predicted_sentiment: np.ndarray,
    ) -> dict[str, object]:
        per_emotion = {}
        for index, label in enumerate(EMOTIONS):
            precision, recall, f1, _ = precision_recall_fscore_support(
                actual_emotions[:, index],
                predicted_emotions[:, index],
                average="binary",
                zero_division=0,
            )
            per_emotion[label] = {
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "support": int(actual_emotions[:, index].sum()),
            }
        return {
            "emotion_micro_f1": float(
                f1_score(actual_emotions, predicted_emotions, average="micro", zero_division=0)
            ),
            "emotion_macro_f1": float(
                f1_score(actual_emotions, predicted_emotions, average="macro", zero_division=0)
            ),
            "emotion_hamming_loss": float(hamming_loss(actual_emotions, predicted_emotions)),
            "per_emotion": per_emotion,
            "sentiment_accuracy": float(np.mean(actual_sentiment == predicted_sentiment)),
            "sentiment_macro_f1": float(
                f1_score(actual_sentiment, predicted_sentiment, average="macro", zero_division=0)
            ),
            "sentiment_confusion_matrix": confusion_matrix(
                actual_sentiment, predicted_sentiment, labels=range(len(SENTIMENTS))
            ).tolist(),
        }

    def _gated_probabilities(
        self, rows: list[FeatureRow], model_row: object
    ) -> tuple[np.ndarray, np.ndarray]:
        import torch

        from app.emotion_fusion import GatedEmotionFusion

        payload = torch.load(model_row["artifact_path"], map_location="cpu", weights_only=True)
        model = GatedEmotionFusion(payload["text_dim"], payload["image_dim"], hidden_dim=64)
        model.load_state_dict(payload["state"])
        model.eval()
        emotions, sentiments = [], []
        with torch.inference_mode():
            for row in rows:
                emotion_logits, sentiment_logits = model(
                    torch.tensor(row.text, dtype=torch.float32)[None],
                    torch.tensor(row.image, dtype=torch.float32)[None],
                )
                emotions.append(torch.sigmoid(emotion_logits)[0].numpy())
                sentiments.append(torch.softmax(sentiment_logits, dim=-1)[0].numpy())
        return np.asarray(emotions), np.asarray(sentiments)

    def evaluate_gated(
        self, feature_version: str = PRETRAINED_FEATURE_VERSION
    ) -> dict[str, object]:
        rows = self.rows(feature_version)
        validation = [row for row in rows if row.split == "validation"]
        test = [row for row in rows if row.split == "test"]
        if not validation or not test:
            return {
                "available": False,
                "reason": "Validation and held-out test embeddings are both required",
                "validation_samples": len(validation),
                "test_samples": len(test),
            }
        model_row = self.connection.execute(
            "SELECT * FROM emotion_models WHERE model_name='emotion-fusion' AND feature_version=? ORDER BY id DESC LIMIT 1",
            (feature_version,),
        ).fetchone()
        if not model_row:
            return {"available": False, "reason": "No compatible gated checkpoint is registered"}
        validation_probabilities, _ = self._gated_probabilities(validation, model_row)
        thresholds = tune_thresholds(
            validation_probabilities, np.asarray([row.emotions for row in validation])
        )
        test_probabilities, test_sentiment_probabilities = self._gated_probabilities(
            test, model_row
        )
        actual_emotions = np.asarray([row.emotions for row in test])
        actual_sentiment = np.asarray([row.sentiment for row in test])
        report = self._metric_report(
            actual_emotions,
            test_probabilities >= thresholds,
            actual_sentiment,
            np.argmax(test_sentiment_probabilities, axis=1),
        )
        return {
            "available": True,
            "model_version": model_row["model_version"],
            "feature_version": feature_version,
            "validation_samples": len(validation),
            "test_samples": len(test),
            "thresholds": dict(zip(EMOTIONS, thresholds.astype(float).tolist(), strict=True)),
            **report,
            "warning": "This 200-image research sample is a functional evaluation, not a benchmark.",
        }

    def evaluate_artifact(
        self, path: Path, rows: list[FeatureRow] | None = None
    ) -> dict[str, object]:
        payload = pickle.loads(path.read_bytes())
        rows = rows or self.rows()
        test = [row for row in rows if row.split == "test"] or rows
        actual_emotions = np.array([row.emotions for row in test])
        actual_sentiment = np.array([row.sentiment for row in test])
        if payload["model_name"] == "frequency-baseline":
            predicted_emotions = np.tile(payload["emotion_rates"] >= 0.5, (len(test), 1))
            predicted_sentiment = np.full(len(test), payload["sentiment_label"])
        else:
            values = self._matrix(test, payload["model_name"])
            values = payload["scaler"].transform(values)
            predicted_emotions = payload["emotion"].predict(values)
            predicted_sentiment = payload["sentiment"].predict(values)
        return {
            "samples": len(test),
            **self._metric_report(
                actual_emotions, predicted_emotions, actual_sentiment, predicted_sentiment
            ),
            "warning": "Smoke-subset functional result; not a benchmark.",
        }
