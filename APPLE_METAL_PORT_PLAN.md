# CogKit → Apple Metal (MPS) Support — Port Plan

**Status:** plan, ready to execute · **Branch:** create `apple-silicon-mps` off `main`
**Executor:** a fresh Fable · **Plan author:** Claude (Opus 4.8) · **Date:** 2026-07-08

This is an executable spec. It assumes the reader is comfortable in the CogKit codebase
(read the repo-root `CLAUDE.md` first) and has done Apple-Silicon/MPS work before. **Line
numbers cite a snapshot and drift — re-grep the symbol before editing.** This port is the
twin of the finetrainers MPS port; that plan and its lessons are the primary prior art.

---

## Scoping decisions

| Decision        | Value                                                    | Consequence                                                                                                                                 |
| --------------- | -------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| Target hardware | M-series **64 GB+** (Max/Ultra)                          | Memory is tight but workable; lean on the existing precompute-caching + `UNLOAD_LIST` offload rather than new offload machinery in phase 1. |
| First model     | **`cogview4-6b` LoRA** — ✅ confirmed by Eric 2026-07-08 | Image path: no video decode, simplest to CPU-verify. Fallback if memory forces it: `cogvideox-t2v` 2B LoRA.                                 |
| Goal            | **Correctness first**                                    | One model training _correctly_ on MPS end-to-end, verified against a CPU oracle. Speed/memory tuning is an explicit later phase.            |
| Parallelism     | **Single-process, `world_size=1`**                       | Do **not** try to run FSDP/NCCL on MPS. Carve out a clean single-device lane; the `strategy: "DDP"` path is the closest existing hook.      |

---

## §0 — Required reading & house context (reference; do not re-derive)

- **Repo `CLAUDE.md`** (root) — the trainer registry, the torchrun launch path, the
  Config/Component/State triad, the `.cache/` footgun. **Read first.**
- **finetrainers PORT_PLAN** — `/Users/eeaglstun/Documents/dev/finetrainers/docs/apple_silicon/PORT_PLAN.md`.
  The sibling port. Same six-problems framing, same correctness-first discipline. Mine it
  for the _method_; the code is different.
- **Eric's MPS-porting playbook (authoritative):**
  `https://ai.ericeaglstun.com/deep-dives/porting-ml-to-apple-silicon/` — the six-problems
  framework this plan is organized around (memory `apple-silicon-porting-deepdive`).
- **`apple-silicon` skill** (`~/.claude/skills/apple-silicon/`) — Metal/MPS reference shelf.
  ⚠️ **Relevance caveat:** it was built for CTranslate2's **hand-written C++/MSL** Metal
  engine. **We are NOT writing Metal kernels here** — CogKit rides PyTorch's `torch.mps`
  backend. Reuse the _discipline_ (parity testing, fp16/bf16 "confident garbage" awareness),
  **not** the kernels/MPSMatrixMultiplication/autorelease-pool material.
- **`ai-dev` skill** — MLX glossary entry. MLX is **not** a drop-in backend; CogKit is
  welded to PyTorch + diffusers + peft. Out of scope. Mention only as a hypothetical.

---

## §1 — Architecture reality (why the plan is shaped this way)

CogKit's finetune stack is built around **distributed CUDA**: every run — even single-GPU —
goes through `torchrun`, `dist.init_process_group(backend="nccl")`, and an FSDP/DDP wrap.
Apple Silicon is a **single unified-memory device**; NCCL is CUDA-only and FSDP on MPS is
effectively unsupported.

**Therefore the port = carve out a clean single-process MPS lane** at `world_size=1` with
**no sharding**. Unlike finetrainers, CogKit has **no pre-existing single-device branch** —
the distributed path is mandatory (`BaseTrainer.__init__` → `_init_distributed`). So this
port does slightly more surgery than finetrainers did, but the surface is small and
concentrated in `finetune/base/base_trainer.py`, `finetune/utils/dist.py`, and
`finetune/utils/memory.py`.

Good news: the heavy machinery is friendly to single-device. Precompute caching means the
big text encoder + VAE only run during dataset preprocessing, then unload (per each
trainer's `UNLOAD_LIST`). DCP checkpointing works over a `gloo` process group at ws=1.

The nailed-shut doors are all device/backend hardcodes, inventoried next.

---

## §2 — Blocker inventory (verified during survey; re-grep before editing)

| #   | Six-problems     | Location                                                                                                                             | Problem                                                                                                                                                                                                                                                                 | Fix                                                                                                                                                                                                                                                                                                                                          |
| --- | ---------------- | ------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | Device selection | `finetune/utils/dist.py:29` `get_device()`                                                                                           | Returns `torch.device(f"cuda:{get_local_rank()}")` — **hardcoded cuda**. This is THE device chokepoint; everything routes through it.                                                                                                                                   | Device-aware: `mps` when `torch.backends.mps.is_available()`, else `cuda:{rank}`, else `cpu`. Add a `COGKIT_DEVICE` env escape hatch here.                                                                                                                                                                                                   |
| 1   | Device selection | `finetune/base/base_trainer.py:105`                                                                                                  | `dist.init_process_group(backend="nccl", ...)` hardcoded.                                                                                                                                                                                                               | `gloo` on MPS/CPU, `nccl` on CUDA.                                                                                                                                                                                                                                                                                                           |
| 1   | Device selection | `finetune/base/base_trainer.py:106`                                                                                                  | `torch.cuda.set_device(get_local_rank())` — **`torch.mps` has no `set_device`**; crashes on Mac.                                                                                                                                                                        | Guard: only call on CUDA.                                                                                                                                                                                                                                                                                                                    |
| 1   | Device selection | `finetune/utils/memory.py:13-17,34-35`                                                                                               | `get_memory_statistics` + `free_memory` are 100% `torch.cuda.*` (`current_device`, `memory_allocated`, `empty_cache`, `ipc_collect`). Note: also **ignores its `device` arg** today.                                                                                    | Device-branch: on MPS use `torch.mps.current_allocated_memory()` / `torch.mps.empty_cache()` (no `ipc_collect`, no `reset_peak_memory_stats`); return `None`/0 for unavailable stats.                                                                                                                                                        |
| 1   | Device selection | `finetune/diffusion/trainer.py:305`                                                                                                  | `torch.cuda.reset_peak_memory_stats(self.state.device)` — no MPS equivalent.                                                                                                                                                                                            | Guard behind `if is_cuda`.                                                                                                                                                                                                                                                                                                                   |
| —   | Parallelism      | `finetune/base/base_trainer.py:180-226` `prepare_model`                                                                              | For any strategy != `DDP` → **FSDP wrap** (unsupported on MPS). The `DDP` branch wraps in `DistributedDataParallel` (needs a process group; works over gloo at ws=1 but is pointless overhead) and relies on `.no_sync()` (L357) + `unwrap_model`→`.module` (L458-461). | Add a **single-device lane**: when device is MPS (or a new `strategy: "SINGLE"`), skip FSDP/DDP entirely — just `self.components.transformer.to(device)`. Then `no_sync()` must become a `nullcontext()` and `unwrap_model` must return the bare module. Verify `clip_grad_norm_` uses the plain-`torch.nn.utils` branch (L370), not FSDP's. |
| 4   | CUDA-only deps   | `.../cogview4/lora_trainer.py`, `.../cogvideox_t2v/lora_trainer.py`, `.../cogvideox_i2v/lora_trainer.py` (grep `BitsAndBytesConfig`) | QLoRA `low_vram: true` path uses **bitsandbytes NF4** — CUDA-only.                                                                                                                                                                                                      | On MPS: if `low_vram: true`, **hard error** with a clear message pointing at bf16/`low_vram: false`. Don't silently ignore.                                                                                                                                                                                                                  |
| 3   | Float precision  | `mixed_precision` config (`fp16`/`bf16`) + FlowMatch scheduler + GLM/T5 encoders                                                     | bf16/fp16 on MPS produce **silently-wrong numbers, not crashes** ("confident garbage").                                                                                                                                                                                 | This is the correctness risk. Verify every stage against a CPU oracle (§4). If a stage diverges, force that stage to fp32 on MPS.                                                                                                                                                                                                            |
| 2   | Missing MPS ops  | env / launch                                                                                                                         | Unsupported ops crash instead of falling back.                                                                                                                                                                                                                          | Ship `PYTORCH_ENABLE_MPS_FALLBACK=1` in the MPS launch script; warn at startup if unset on MPS.                                                                                                                                                                                                                                              |
| —   | Launch           | `quickstart/scripts/*/start_train.sh`                                                                                                | `torchrun --nproc_per_node=[N GPUs]` assumes CUDA GPUs.                                                                                                                                                                                                                 | Add an MPS recipe: `torchrun --nproc_per_node=1 --master_port=29501 ../train.py --yaml config.yaml` with the env vars set. `LOCAL_RANK`/`WORLD_SIZE` are set by torchrun even at ws=1, so `get_local_rank()` still works.                                                                                                                    |

Also grep before finishing: any other `torch.cuda.` call site, and any `.cuda()` /
`device="cuda"` string literal in the datasets and `python/generation/` inference path
(inference is a separate, simpler surface — see §5).

---

## §3 — Phased implementation

### Phase 1 — Unblock a single-device MPS training run _(the only phase that gates a first run)_

Do these in order; each is small:

1. **`utils/dist.py::get_device()`** — device-aware + `COGKIT_DEVICE` env escape hatch. Single chokepoint.
2. **`base_trainer.py::_init_distributed`** — gloo on non-CUDA; guard `torch.cuda.set_device`.
3. **`base_trainer.py::prepare_model`** — single-device lane (no FSDP/DDP wrap on MPS).
4. **`base_trainer.py`** — make `no_sync()` (L357) and `unwrap_model` (L458) correct for the bare-module case.
5. **`utils/memory.py`** + **`diffusion/trainer.py:305`** — device-branch the memory/stat calls.
6. **QLoRA guard** — hard error on `low_vram: true` + MPS in the affected `lora_trainer.py` files.
7. **Launch** — `quickstart/scripts/<task>/start_train_mps.sh` with `PYTORCH_ENABLE_MPS_FALLBACK=1`, `world_size=1`.
8. Pick the first model's config, set `strategy` to the single-device value, `mixed_precision: bf16`, tiny dataset (use `quickstart/data/<task>/`), `train_epochs: 1`, `checkpointing_steps` small.

**Exit criterion:** the run reaches `train()` and completes ≥1 optimizer step + 1
checkpoint on MPS without crashing. Numerical correctness is Phase 2 — don't trust the loss yet.

### Phase 2 — Correctness (CPU-parity verification)

The whole point. See §4 for the methodology. Exit criterion: MPS matches CPU oracle within
tolerance for the compute_loss forward, and a short MPS LoRA run shows a **decreasing,
non-NaN** loss curve that tracks a CPU run of the same seed/data.

### Phase 3 — Inference on MPS

Wire the `cogkit inference` / `python/generation` path to MPS (`load_type`/offload choices
differ — there's no `cpu_model_offload`-to-CUDA assumption to satisfy). Lower risk than
training; can be done in parallel. See §5.

### Phase 4+ (out of scope now)

Memory optimization, packing on MPS, multi-model generalization, speed. Do **not** start
these until Phases 1–2 are green and Eric signs off.

---

## §4 — Correctness methodology (mirror finetrainers / the CT2 op-parity discipline)

CogKit conveniently precomputes latents + prompt embeddings into `data_root/.cache/`. Use
that as the parity fixture:

1. **Same seed, same cached batch, two devices.** Run `compute_loss(batch)` for the chosen
   trainer on **CPU** (`COGKIT_DEVICE=cpu`) and on **MPS**, from the identical cached batch
   and `seed`. Compare the loss and, ideally, the pre-loss `noise_pred` tensor within a
   per-dtype tolerance (fp32 tight, bf16 loose). CPU is the oracle.
2. **Stage-isolate on divergence.** If loss diverges, walk the forward: VAE encode → text
   embed → `add_noise` → transformer forward. Diff each stage CPU-vs-MPS to find the first
   one that drifts. The prime suspects on MPS: bf16 accumulation, the FlowMatch
   `get_sigmas`/`add_noise` math, and the attention processor.
3. **fp32 fallback per-stage**, not globally, when a stage is the culprit — keep bf16
   everywhere it's proven safe.
4. **⚠️ "Confident garbage" rule:** on MPS, bf16/fp16 wrongness is silent. A run that
   completes and shows a plausible loss is **not** evidence of correctness. Only CPU-parity is.
5. **`.cache/` hygiene:** if you change resolution/packing/dataset, delete `data_root/.cache/`
   (and the test split's) — CogKit does not auto-invalidate it (repo `CLAUDE.md`).

Add these as a small `tests/` module (CogKit already uses pytest; note pytest isn't in the
declared deps — install it). Follow the existing `tests/test_sampler.py` shape.

---

## §5 — Inference path (Phase 3 detail)

Separate, simpler surface from training. Entry points: `cogkit inference` (CLI) →
`cogkit/python/generation/{image,video}.py`, and `load_pipeline` / offload helpers in
`cogkit/utils/`. The CLI's `--load_type` today is `{cuda, cpu_model_offload,
sequential_cpu_offload}` — add/enable an `mps` path and make the default offload choice
sane for unified memory. Grep `cogkit/python` and `cogkit/utils` for `"cuda"` / `.cuda()`
/ `device_map` assumptions. No distributed concerns here.

---

## §6 — Definition of done (Phase 1–2)

- [ ] First model (locked above) completes a real LoRA run on MPS: ≥1 epoch on the
      `quickstart/data/` sample set, checkpoints written, `merge.py` produces a loadable adapter.
- [ ] CPU-parity test passes for `compute_loss` within documented tolerance.
- [ ] MPS loss curve tracks the CPU run (same seed/data), non-NaN, decreasing.
- [ ] QLoRA / FSDP / packing on MPS fail **loudly** with actionable messages (not silently).
- [ ] `pre-commit run --all-files` clean; no `torch.cuda.*` reachable on the MPS path.
- [ ] A short `docs/`-**excluded** note (or an update to repo `CLAUDE.md`) documenting the
      MPS lane, the launch script, and the correctness caveats. (Remember: anything under
      `docs/**` auto-publishes to the public Docusaurus site — keep internal notes out of it.)

---

## Appendix — Environment quickstart (Mac)

```bash
git checkout -b apple-silicon-mps
export PYTORCH_ENABLE_MPS_FALLBACK=1      # missing-op CPU fallback
export COGKIT_DEVICE=mps                  # (new escape hatch added in Phase 1)
# CPU oracle run for parity:  COGKIT_DEVICE=cpu

# training is torchrun-only, even at world_size=1:
cd quickstart/scripts/<task>
torchrun --nproc_per_node=1 --master_port=29501 ../train.py --yaml config.yaml
```

Deps: PDM project, Python ≥3.10, PyTorch with MPS (a recent 2.x), `diffusers` from **git
main** (not PyPI), the `finetune` extra for training. `bitsandbytes` is CUDA-only — expect
it absent/unused on Mac (that's the QLoRA guard's whole point).
