import numpy as np
import pytest
import torch

from app.emotion import EMOTIONS, EmotionService, FeatureRow


def _rows(text_dim: int = 4, image_dim: int = 5) -> list[FeatureRow]:
    rows = []
    for index, split in enumerate(("train", "train", "train", "train", "validation", "validation")):
        emotions = np.zeros(len(EMOTIONS), dtype=np.int8)
        emotions[index % 2] = 1
        rows.append(
            FeatureRow(
                meme_id=index + 1,
                text=np.full(text_dim, index + 1, dtype=np.float32),
                image=np.full(image_dim, index + 1, dtype=np.float32),
                emotions=emotions,
                sentiment=index % 3,
                split=split,
            )
        )
    return rows


def _service(tmp_path, monkeypatch, rows: list[FeatureRow]) -> EmotionService:
    service = EmotionService(tmp_path / "training.db")
    monkeypatch.setattr(service, "rows", lambda _version: rows)
    return service


def test_gated_checkpoint_resume_restores_optimizer_and_advances_epoch(
    tmp_path, monkeypatch
) -> None:
    service = _service(tmp_path, monkeypatch, _rows())
    try:
        first = service.train_gated(tmp_path / "artifacts", epochs=1)
        first_payload = torch.load(first["checkpoint"], map_location="cpu", weights_only=True)
        resumed = service.train_gated(
            tmp_path / "artifacts", epochs=1, resume_from=first["checkpoint"]
        )
        resumed_payload = torch.load(resumed["checkpoint"], map_location="cpu", weights_only=True)
    finally:
        service.close()

    assert first_payload["optimizer_state"]["state"]
    assert resumed["resumed"] is True
    assert resumed_payload["epoch"] == first_payload["epoch"] + 1
    assert resumed_payload["optimizer_state"]["state"]


def test_gated_checkpoint_rejects_incompatible_feature_dimensions(tmp_path, monkeypatch) -> None:
    service = _service(tmp_path, monkeypatch, _rows())
    try:
        first = service.train_gated(tmp_path / "artifacts", epochs=1)
        monkeypatch.setattr(service, "rows", lambda _version: _rows(text_dim=6))
        with pytest.raises(ValueError, match="feature dimensions"):
            service.train_gated(tmp_path / "artifacts", epochs=1, resume_from=first["checkpoint"])
    finally:
        service.close()


def test_gated_weights_handle_absent_classes_and_validate_inputs(tmp_path, monkeypatch) -> None:
    service = _service(tmp_path, monkeypatch, _rows())
    try:
        result = service.train_gated(
            tmp_path / "artifacts",
            epochs=1,
            emotion_loss_weight=2.0,
            sentiment_loss_weight=0.5,
        )
        payload = torch.load(result["checkpoint"], map_location="cpu", weights_only=True)
        configuration = payload["training_configuration"]
        assert configuration["emotion_pos_weights"][2:] == [1.0] * 4
        assert configuration["emotion_loss_weight"] == 2.0
        assert configuration["sentiment_loss_weight"] == 0.5
        with pytest.raises(ValueError, match="six positive"):
            service.train_gated(tmp_path / "artifacts", epochs=1, emotion_pos_weights=[1.0] * 5)
        with pytest.raises(ValueError, match="six positive"):
            service.train_gated(
                tmp_path / "artifacts", epochs=1, emotion_pos_weights=[1.0, 1.0, 0, 1.0, 1.0, 1.0]
            )
    finally:
        service.close()
