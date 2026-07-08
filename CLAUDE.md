# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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
(bitsandbytes NF4 quantization) and forces the DDP path; its checkpoints are saved
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

## Conventions

- **ruff** (line length 100, double quotes) + **mypy** are enforced via pre-commit;
  run `pre-commit run --all-files` before pushing — CI runs exactly this.
- Version is derived from git tags via `hatch-vcs` (`src/cogkit/_version.py` is
  generated — do not edit or commit it manually).
- Follow the existing `@override` + abstractmethod pattern when adding trainers; keep
  the module-bottom `register(...)` call so auto-discovery picks it up.
