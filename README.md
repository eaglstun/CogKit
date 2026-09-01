# CogKit

## Introduction

CogKit is an open-source project that provides a user-friendly interface for researchers and developers to utilize models from ZhipuAI, currently supports [CogView](https://huggingface.co/collections/THUDM/cogview-67ac3f241eefad2af015669b) (image generation) and [CogVideoX](https://huggingface.co/collections/THUDM/cogvideo-66c08e62f1685a3ade464cce) (video generation) series. Users must comply with legal and ethical guidelines to ensure responsible implementation.

Visit our [Docs](https://thudm.github.io/CogKit) to start.

## Features

- Training Optimization: Includes pre-computation and caching of latents and embeddings, sequence packing, and various memory-efficient strategies to improve training throughput and reduce GPU memory usage.

- Native Resolution Training Support: Seamlessly train models at original image resolutions for improved quality and consistency.

- Easy-to-use Interface: Offers multiple easy-to-use inference options, including a CLI, OpenAI-compatible API server, and interactive Gradio-based UIs for both training and inference.

## Apple Silicon (MPS) — this fork

This fork adds a single-device Apple Silicon training lane (verified on an M4 Max 64GB:
cogview4-6b LoRA trains correctly on MPS, CPU-parity checked). Plan/history in
`APPLE_METAL_PORT_PLAN.md`; working notes in `CLAUDE.md`.

```bash
# setup (uv; bitsandbytes is skipped automatically on macOS)
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch torchvision
uv pip install --python .venv/bin/python -e ".[finetune]" pytest

# train (t2i, cogview4-6b LoRA)
cd quickstart/scripts/t2i
bash start_train_mps.sh          # torchrun ws=1, strategy: SINGLE, bf16

# verify numerics against the CPU oracle (~6 min) + overfit check
.venv/bin/python -m pytest tests/test_mps_cpu_parity.py -q -s
```

CogView4 inference supports MPS through model offload (recommended) or direct unified-memory
placement:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 cogkit inference "a prompt" THUDM/CogView4-6B \
  --load_type cpu_model_offload --output_file out.png

# Higher startup residency; useful for explicit full-pipeline placement experiments.
PYTORCH_ENABLE_MPS_FALLBACK=1 cogkit inference "a prompt" THUDM/CogView4-6B \
  --load_type mps --output_file out.png
```

Not supported on MPS: QLoRA (`low_vram`), FSDP strategies, and video (t2v/i2v) training —
all fail loudly with pointers. CogVideo inference has not yet been validated on MPS.

## Roadmap

- [ ] Add support for CogView4 ControlNet model
- [ ] Docker for easy deployment

## License

This project is licensed under the [Apache 2.0 License](LICENSE).
