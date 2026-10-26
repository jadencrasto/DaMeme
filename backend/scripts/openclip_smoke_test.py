"""Load the authorized OpenCLIP checkpoint and run one CUDA text/image smoke test."""

import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import open_clip
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "data" / "retrieval" / "catalogs" / "memematch_1000_v2"
OUTPUT = ROOT / "backend" / "benchmarks" / "phase7r_v1_openclip_smoke_test"
MODEL = "ViT-B-32-quickgelu"
PRETRAINED = "openai"


def normalized(values: torch.Tensor) -> torch.Tensor:
    return values / values.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def checkpoint_files() -> list[dict[str, str | int]]:
    cache = Path.home() / ".cache" / "huggingface" / "hub"
    results = []
    for path in [cache / "clip" / "ViT-B-32.pt", Path.home() / ".cache" / "clip" / "ViT-B-32.pt"]:
        if path.is_file():
            results.append(
                {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    return results


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    image_path = sorted((CATALOG / "images").iterdir())[0]
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL, pretrained=PRETRAINED, device="cuda"
    )
    tokenizer = open_clip.get_tokenizer(MODEL)
    load_ms = (time.perf_counter() - started) * 1000
    model.eval()
    with Image.open(image_path) as source:
        image = preprocess(source.convert("RGB")).unsqueeze(0).to("cuda")
    text = tokenizer(["a reaction meme about being surprised"]).to("cuda")
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        text_vector = normalized(model.encode_text(text))
        image_vector = normalized(model.encode_image(image))
    torch.cuda.synchronize()
    inference_ms = (time.perf_counter() - started) * 1000
    similarity = float((text_vector @ image_vector.T).item())
    report = {
        "format_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "python": __import__("sys").version,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "open_clip": getattr(open_clip, "__version__", "2.32.0"),
        "model": MODEL,
        "pretrained": PRETRAINED,
        "image": str(image_path.relative_to(ROOT)),
        "model_load_success": True,
        "text_shape": list(text_vector.shape),
        "image_shape": list(image_vector.shape),
        "text_finite": bool(torch.isfinite(text_vector).all()),
        "image_finite": bool(torch.isfinite(image_vector).all()),
        "text_l2_norm": float(text_vector.norm().item()),
        "image_l2_norm": float(image_vector.norm().item()),
        "text_image_similarity": similarity,
        "model_load_ms": load_ms,
        "inference_ms": inference_ms,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "checkpoint_files": checkpoint_files(),
    }
    (OUTPUT / "smoke_test.report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
