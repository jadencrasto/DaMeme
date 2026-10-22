# Optional model setup

No model package or weight is installed/downloaded by default. Install CPU-capable dependencies only when ready to run real inference:

```powershell
Set-Location backend
uv add --group dev easyocr transformers torch
```

The first EasyOCR/BLIP invocation downloads its own weights into the user cache. Use CPU first; CUDA requires a compatible user-installed PyTorch build and enough free VRAM. Do not load both models concurrently on the 4 GB GPU. Real-model quality has not been validated in this repository; tests use mock providers.
