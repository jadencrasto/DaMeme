import torch

from app.emotion_fusion import GatedEmotionFusion


def test_gated_fusion_forward_and_backward() -> None:
    model = GatedEmotionFusion(8, 12, hidden_dim=16)
    emotions, sentiment = model(torch.randn(3, 8), torch.randn(3, 12))
    loss = torch.nn.BCEWithLogitsLoss()(emotions, torch.zeros(3, 6)) + torch.nn.CrossEntropyLoss()(
        sentiment, torch.zeros(3, dtype=torch.long)
    )
    loss.backward()
    assert emotions.shape == (3, 6)
    assert sentiment.shape == (3, 3)
