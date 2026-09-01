---
---

# Apple Silicon (MPS)

This fork includes a single-device Apple Silicon lane for CogView4 training and inference.
It uses PyTorch's MPS backend; it does not contain custom Metal kernels.

## Supported Today

- CogView4-6B LoRA training with `strategy: "SINGLE"`
- CogView4 image inference with model CPU offload or direct MPS placement
- CPU-to-MPS numerical parity tests for training

QLoRA (`low_vram: true`) and FSDP are not supported on MPS. CogVideo training is also
unsupported because the current macOS video-decoding dependency path is unavailable.
CogVideo inference has not yet been validated on MPS.

## Setup

Use the tested Python 3.12 environment. Python 3.14 is not currently supported by the full
inference dependency set.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch torchvision
uv pip install --python .venv/bin/python -e ".[finetune]" pytest
```

The project conditionally excludes `bitsandbytes` on macOS because it is only used by the
CUDA QLoRA path.

## Image Inference

Model CPU offload is the recommended starting point because it lowers peak accelerator
residency:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 cogkit inference \
  "a watercolor painting of mountains at sunrise" \
  THUDM/CogView4-6B \
  --load_type cpu_model_offload \
  --output_file output.png
```

Direct unified-memory placement is available for experiments:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 cogkit inference \
  "a watercolor painting of mountains at sunrise" \
  THUDM/CogView4-6B \
  --load_type mps \
  --output_file output.png
```

`PYTORCH_ENABLE_MPS_FALLBACK=1` lets operations without an MPS implementation run on CPU.
Such fallbacks can reduce performance, so successful generation is not itself a performance
claim.

## LoRA Training

The MPS recipe still launches one process through `torchrun`; CogKit's checkpointing and
rank helpers depend on an initialized `gloo` process group.

```bash
cd quickstart/scripts/t2i
bash start_train_mps.sh
```

The supplied `config_mps.yaml` uses `strategy: "SINGLE"`, `low_vram: false`, in-process data
loading (`num_workers: 0`), and no pinned CUDA memory. Set `COGKIT_DEVICE=cpu` before launching
the script to run the same lane on the CPU oracle.

## Correctness Checks

MPS reduced-precision computations can finish without an error while producing incorrect
numbers. Run the CPU-to-MPS parity suite when changing the training computation:

```bash
.venv/bin/python -m pytest tests/test_mps_cpu_parity.py -q -s
```

Real-model inference smoke and parity tests are opt-in because they load cached model
weights and are expensive. See `tests/test_inference_mps.py` and the phase history in
`APPLE_METAL_PORT_PLAN.md` before running them.
