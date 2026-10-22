import csv
import io
import zipfile
from pathlib import Path

from PIL import Image

from app.datasets.mood import extract_mood_sample


def test_extract_mood_sample_preserves_source_annotations(tmp_path: Path) -> None:
    rows = [
        {
            "image": f"{index}.png",
            **{
                emotion: emotion if emotion == selected else f"not_{emotion}"
                for emotion in ("fear", "anger", "joy", "sadness", "surprise", "disgust")
            },
            "sentiment": "positive",
        }
        for index, selected in enumerate(("fear", "anger", "joy"))
    ]
    nested = io.BytesIO()
    with zipfile.ZipFile(nested, "w") as images:
        for row in rows:
            image = io.BytesIO()
            Image.new("RGB", (10, 10), "red").save(image, format="PNG")
            images.writestr(f"final/{row['image']}", image.getvalue())
    archive = tmp_path / "mood.zip"
    with zipfile.ZipFile(archive, "w") as outer:
        csv_buffer = io.StringIO()
        writer = csv.DictWriter(csv_buffer, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
        outer.writestr("meme_emotion/final.csv", csv_buffer.getvalue())
        outer.writestr("meme_emotion/final.zip", nested.getvalue())
    output, metadata = tmp_path / "images", tmp_path / "metadata.json"
    assert extract_mood_sample(archive, output, metadata, limit=3) == 3
    saved = __import__("json").loads(metadata.read_text(encoding="utf-8"))
    assert set(saved) == {"0.png", "1.png", "2.png"}
    assert saved["0.png"]["annotations"]["emotions"]["fear"] == "fear"
