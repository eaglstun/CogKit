# AGENTS.md

This file provides guidance to AI tools when working with code in this repository.

## What this is

CogKit is a toolkit (from ZhipuAI/THUDM) for **inference and finetuning** of the
**CogView** (image generation) and **CogVideoX** (video generation) diffusion model
families. It ships three surfaces over the same core: a CLI, an OpenAI-compatible
FastAPI server, and Gradio UIs. The Python package lives in `src/cogkit/`; runnable
training configs, example datasets, and launch scripts live in `quickstart/`.

## Commands

Dependency management is **PDM** (`pdm.lock`, `pyproject.toml`), Python **3.10+**.

```bash
# Lint / format / typecheck (these are what CI runs, via pre-commit)
pdm run lint                 # ruff check
mypy --non-interactive src/cogkit tests   # (or: pdm run typecheck)
pre-commit run --all-files   # exactly what .github/workflows/python-lint.yml runs

# Tests (pytest — not declared in pyproject deps, install it separately)
pytest                       # all tests
pytest tests/test_sampler.py                       # one file
pytest tests/test_sampler.py::test_initialization  # one test

# Inference (CLI). Task (t2i/t2v/i2v/ct2i) is auto-detected from the pipeline + whether --image_file is passed.
cogkit inference "a prompt" THUDM/CogView4-6B --output_file out.png
cogkit inference "a prompt" <model> --lora_model_id_or_path <path>   # apply a LoRA
cogkit launch --host 0.0.0.0 --port 8000            # OpenAI-compatible API server

# Gradio UIs
python gradio/gradio_ui.py   # tabbed Inference + Train UI
```

Install extras: base install is inference-only; **finetuning needs the `finetune`
extra** (`pip install "cogkit[finetune]..."`) which pulls in `datasets`, `wandb`,
`av`, `bitsandbytes`. `diffusers` is pinned to **git main**, not a release — model
classes like `CogView4Transformer2DModel` may not exist in any PyPI diffusers.

## Training: how to launch

Training does **not** go through the `cogkit` CLI. It is launched with `torchrun`
(distributed is mandatory — even single-GPU runs go through the distributed code
path and `dist.init_process_group`). The flow:

1. Pick a task dir under `quickstart/scripts/{t2i,t2v,i2v}/`.
2. Edit its `config.yaml` (this YAML **is** the full arg set — see `BaseArgs`/`DiffusionArgs`).
3. `bash start_train.sh` — which runs `torchrun ... ../train.py --yaml config.yaml`.

`quickstart/scripts/train.py` reads `model_name` + `training_type` + `enable_packing`
from the YAML, calls `get_model_cls(...)` to resolve the trainer class, instantiates
it with the YAML path, and calls `.fit()`.

After LoRA/FSDP training, distributed checkpoints must be merged before use:
`python tools/converters/merge.py --checkpoint_dir <ckpt> --output_dir <out> [--lora]`.
(QLoRA / `low_vram` runs save a plain LoRA and skip this step.) Then pass the output
to `--lora_model_id_or_path` (LoRA) or `--transformer_path` (SFT/FSDP) at inference.

## Architecture

### Trainer registry (the central dispatch mechanism)

Every trainer is registered by `(model_name, training_type)` into the global
`SUPPORTED_MODELS` dict in `src/cogkit/finetune/_register.py`. Registration is a
side effect: `src/cogkit/finetune/diffusion/models/__init__.py` **auto-imports every
`.py` under `models/`** on package import, and each model module ends with a
`register("cogview4-6b", "lora", Cogview4Trainer)` call. So:

- To add a model/training-type, drop a module under `.../models/<family>/<model>/`
  that subclasses the family trainer and calls `register(...)` at module bottom.
- `use_packing=True` maps to a distinct registry key suffix `"-packing"` (e.g.
  `lora-packing`), which is why packing has its own `*_trainer_packing.py` files.
- `get_model_cls(model_name, training_type, use_packing)` is the only lookup entry point.

### Trainer class hierarchy

`BaseTrainer` (`finetune/base/base_trainer.py`) is an ABC owning the entire training
lifecycle — `fit()` calls, in order: `prepare_models` → `prepare_dataset` →
`prepare_trainable_parameters` → `prepare_model` → `prepare_optimizer` → `train`.
It handles distributed setup, FSDP/DDP wrapping, the train loop, gradient
accumulation/sync, checkpoint save/resume (torch DCP), and validation cadence. The
abstract methods a subclass must fill are `load_components`, `prepare_models`,
`prepare_dataset`, `compute_loss`, `validate`.

`DiffusionTrainer` (`finetune/diffusion/trainer.py`) sits between `BaseTrainer` and
the concrete model trainers, implementing the diffusion-specific dataset/validation
scaffolding. Concrete trainers (e.g. `Cogview4Trainer` in
`.../cogview/cogview4/lora_trainer.py`) implement the model-specific pieces:
`load_components` (pipeline/tokenizer/text_encoder/transformer/vae/scheduler),
`encode_text`, `encode_image`, `collate_fn`, `compute_loss`, `validation_step`.

Key invariant: **only the `transformer` component is trained.** Everything else
(vae, text_encoder) is frozen and should be listed in the trainer's `UNLOAD_LIST`
so it can be offloaded during the train step. LoRA vs SFT is decided by
`training_type` in `prepare_trainable_parameters`.

### Config / component / state triad

Trainers thread three pydantic-style objects (`finetune/base/` and
`finetune/diffusion/schemas/`):

- **Args** (`BaseArgs` → `DiffusionArgs`): parsed from the YAML via
  `parse_from_yaml`; the single source of every knob (strategy, precision,
  `enable_packing`, `low_vram`, checkpointing, validation, etc.).
- **Components** (`BaseComponents` → `DiffusionComponents`): the actual model
  modules (transformer, vae, text_encoder, tokenizer, scheduler, pipeline_cls).
- **State** (`BaseState` → `DiffusionState`): runtime/distributed state (ranks,
  device, weight dtype, resolved train resolution, generator).

### Distributed strategy

`strategy` in the config selects the parallelism: `DDP` or an FSDP sharding mode
(`NO_SHARD`, `SHARD_GRAD_OP`, `FULL_SHARD`, `HYBRID_SHARD`). FSDP uses a
size-based auto-wrap policy (params ≥ 1e8). `low_vram: true` enables **QLoRA**
(bitsandbytes NF4 quantization) and requires `DDP` or `SINGLE`; its checkpoints are saved
as plain LoRA and must **not** be run through `merge.py`.

### Data & performance features

- **Latent/embedding precomputation**: datasets encode images→latents and
  prompts→embeddings once and cache them in a `.cache/` dir inside `data_root`.
  **This cache is not invalidated automatically** — if the dataset changes, or you
  flip `enable_packing`, you must delete `.cache/` manually and retrain.
- **Native-resolution / sequence packing**: `enable_packing: true` trains at each
  image's original resolution instead of resizing to `train_resolution`. It uses a
  custom `NaivePackingSampler` (`finetune/samplers/`) and the `*_packing.py` trainer
  variants.
- Dataset layouts per task (t2i / t2v / i2v) are documented in
  `docs/04-Finetune/03-Data Format.md`; templates live in `quickstart/data/`.

### Inference / API / Gradio

- `src/cogkit/python/generation/{image,video}.py` hold the actual generate functions;
  `cogkit/utils/` provides pipeline loading, LoRA inject/merge, dtype/seed helpers,
  and `guess_generation_mode` (which decides t2i vs i2v vs ct2i from the pipeline
  class and whether an input image was given — see `cogkit/types/generation_mode.py`).
- The API server (`cogkit/api/`) is a FastAPI app (`get_application`) exposing an
  OpenAI-compatible surface under `/v1`; it is configured from env vars
  (`APISettings`, see `.env.template`) and holds an `ImageGenerationService` in app state.
- Gradio lives in `gradio/` and imports from the installed `cogkit` package.

## Apple Silicon (MPS) lane

This fork trains CogView4 and runs CogView4/CogVideoX inference on a single Apple Silicon
device. **Status: training and inference both work; performance work is in progress —
`APPLE_SILICON_PERFORMANCE_PLAN.md` Steps 1–4 are accepted. Step 5's premise (dataloader
overlap) is retired: data wait measures 0.2% of a step.** Each phase has
a plan with an acceptance record (`APPLE_METAL_PORT_PLAN.md` for the training port and MPS
inference, `COGVIDEO_MPS_INFERENCE_PLAN.md` for video, the performance plan above); dated
measurements and raw JSON live in `docs/benchmarks/` — take any number from there.

Env: three venvs, all built with uv (not PDM), and they are **not interchangeable**:

| venv                     | python | torch                | use                                      |
| ------------------------ | ------ | -------------------- | ---------------------------------------- |
| `.venv`                  | 3.12   | pinned 2.12.1        | reference for gates; **no bitsandbytes** |
| `.venv-pytorch-fork`     | 3.12   | local fork           | older fallback lane; **no bitsandbytes** |
| `.venv-pytorch-fork-314` | 3.14   | local fork (2.15.0a) | **primary lane; the only one with bnb**  |

The local PyTorch fork lives at `~/Documents/dev/pytorch` and needs `USE_DISTRIBUTED=1` (the
training lane launches over gloo). **Trap:** `start_train_mps.sh` defaults `TORCHRUN` to
`.venv/bin/torchrun`, which has no bitsandbytes, so a QLoRA launch there dies inside torchrun
with the real error swallowed by a `ChildFailedError`. Pass
`TORCHRUN=$PWD/.venv-pytorch-fork-314/bin/torchrun` for anything `low_vram`. Key facts:

- `strategy: "SINGLE"` = no FSDP/DDP wrap; still launched via **torchrun at
  `--nproc_per_node=1`** over a `gloo` process group (DCP checkpointing and the rank
  helpers depend on it). Launch: `quickstart/scripts/t2i/start_train_mps.sh`
  (`config_mps.yaml`). `COGKIT_DEVICE` env forces the device (`cpu` = parity oracle);
  otherwise auto-detect cuda → mps → cpu in `finetune/utils/dist.py::get_device`.
- **`gradient_checkpointing: false` in `config_mps.yaml` is deliberate, not a leftover** — it
  trades MPS memory for a large step-time win on a 64 GB machine. Set it back to `true` on a
  smaller Mac; the `BaseArgs` default is still `True`, so CUDA is unaffected.
- CogView4 training installs `CogKitCogView4TrainingAttnProcessor`
  (`finetune/diffusion/models/cogview/cogview4/attention.py`) over the diffusers one. Same
  masking semantics, built once per forward instead of per block; packed training still
  delegates upstream. Changes there must keep `tests/test_cogview4_attention.py` green — it
  checks outputs _and_ gradients against the upstream processor.
- FSDP strategies **hard-error on non-CUDA** (`BaseTrainer._check_device_compat`).
  QLoRA (`low_vram: true`) no longer does: the same guard now runs a real 4-bit round trip
  on the target device (`finetune/utils/quantization.py`) and fails only when bitsandbytes
  cannot actually quantize there. Keep it a capability check — a `sys.platform` test would
  both lock out working Metal builds and pass on broken ones. `pyproject.toml` still leaves
  bitsandbytes off the darwin dependency set (no published wheel); Apple Silicon needs a
  source build with Metal kernels. The `BitsAndBytesConfig` in each `lora_trainer.py` must
  stay lazily constructed inside the `low_vram` branch.
- **The Metal bitsandbytes is Eric's fork at `~/Documents/dev/bitsandbytes`** — build with
  `cmake -DCOMPUTE_BACKEND=mps -S . && make -j8` then `uv pip install -e . --no-build-isolation`
  (two steps: `wheel.cmake = false`, so pip never re-runs cmake). Read its
  `docs/apple_silicon/MPS_STATUS.md` before trusting anything about which ops are native.
  **Both directions of the 4-bit matmul are now native for every dtype**, bf16 included
  (`MPSMatrixMultiplication` has no bf16; `MPSGraph` does), and `MatMul4Bit.backward` is fused
  rather than composing a dequant with a torch matmul. Measured on this lane vs the pre-native
  baseline: forward −15.2%, backward −16.2%, step −15.3%. `BNB_MPS_REQUIRE_NATIVE=1` turns
  silent fallbacks into hard failures; `BNB_MPS_DISABLE_BF16_GEMM` / `..._BWD` force the
  fallback per direction, which is how the A/B above was run against one binary.
- Inference placement is idempotent: `before_generation` short-circuits on
  `_cogkit_load_type`, and the API service places the pipeline at startup rather than on the
  first request. VAE slicing/tiling are `vae_slicing`/`vae_tiling` there and a
  `vae_memory_saving` API setting, **defaulting on** -- off is ~17% faster warm but costs
  21 GB of MPS driver allocation (`docs/benchmarks/APPLE_MPS_VAE_DECODE_2026-09-03.md`).
  `None` means _leave the setting alone_: `generate_image`/`generate_video` forward only
  `load_type`, so treating unset as "enable" silently resets an explicit setting every request.
- Video _training_ is unsupported: torchvision ≥0.23 removed `VideoReader`, so the dataset
  modules guard that import (type-annotation use only) and `cv2` is imported lazily. t2v/i2v
  training fails loudly at use-time. Video _inference_ on MPS is a separate lane and works.
- **Correctness rule:** bf16/fp16 on MPS can be silently wrong. A completed run is not
  evidence — only CPU parity is (`tests/test_mps_cpu_parity.py` for training,
  `tests/test_inference_mps.py` for inference; both heavy, both need cached weights + the
  precompute cache). Cheap unit lanes: `tests/test_single_device_lane.py`,
  `tests/test_cogview4_attention.py`.
- **SDPA has no fused MPS backward.** `attention.cpp` routes MPS to the fused kernel only when
  grad is off, so training always runs the composite math path. Do not credit training with an
  inference kernel win.
- Timing on MPS needs wall clock plus an explicit `torch.mps.synchronize()` — `ProfilerActivity.MPS`
  does not exist and `torch.mps.Event` timing is unreliable. See `finetune/utils/performance.py`.
- **Benchmark discipline on this machine** (each of these has produced a wrong published number
  here at least once):
  - _Check the machine is idle first._ `uptime` — a VS Code file-scan storm has put load average
    at 200–298, where a contended probe was not merely noisy but **reversed the winner** of an
    A/B. A 64 MB `clone()` control should read ~0.30 ms (~427 GB/s) on an M4 Max; take it before
    and after every table.
  - _Never run arms in a fixed order._ Reversing arm order moved the same arm's step time by
    −0.50 s to +0.64 s, as large as the effects being measured. Run half the reps reversed.
  - _Get a free error bar from a stage the change cannot touch._ A switch that only affects the
    forward should move the backward by 0%; whatever it does move it by is your noise floor
    (~2–4% here). Report ranges, and treat n=2 as a rumour — an n=2 step delta of −7% with
    "no overlap" became −3.7% with overlap at n=4.
  - _Never compare across torch builds or days._ The 2.15 fork cut the CogVideo warm transformer
    stage from 14.50 s to ~6 s, larger than most effects under test.
- A dead torchrun can leave port 29501 held: `lsof -ti :29501 | xargs kill -9`.

## Conventions

- **ruff** (line length 100, double quotes) + **mypy** are enforced via pre-commit;
  run `pre-commit run --all-files` before pushing — CI runs exactly this.
- Version is derived from git tags via `hatch-vcs` (`src/cogkit/_version.py` is
  generated — do not edit or commit it manually).
- Follow the existing `@override` + abstractmethod pattern when adding trainers; keep
  the module-bottom `register(...)` call so auto-discovery picks it up.
