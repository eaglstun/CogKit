# CogKit Apple Silicon Performance Plan

**Branch:** `perf/apple-silicon-training`  
**Baseline:** CogView4-6B LoRA, 512x512, bf16, batch size 1, approximately 7 s/it on
the documented M4 Max 64 GB run.  
**Rule:** a completed MPS run is not a correctness result. Every numerical change must
retain the CPU-to-MPS parity gates in `tests/test_mps_cpu_parity.py`.

**Current status:** Steps 1-4 are done and recorded below. Step 2 landed the large win
(gradient checkpointing off on MPS: median step 10.46 s -> 7.10 s). Step 3 landed as a
correctness-neutral change with no measurable speedup, and its premise is now retired -- do not
spend more time on attention-mask construction. Step 4 found placement already idempotent and
turned VAE memory saving into a setting: it is worth ~17% of warm latency but costs 21 GB, so the
default stays on. **VAE decode is still 66-78% of warm latency and is still the target; tiling is
just not the lever.** Step 5's premise is also retired (data wait measures 0.2% of a step).
Enable profiling by setting `profile_steps: 3` (or more) in `config_mps.yaml`. Keep at least one
warmup step, and extend the window through a checkpoint when measuring serialization.

This plan deliberately starts with measurement. MPS work is asynchronous, so ordinary
wall-clock timers can report command-encoding time rather than GPU execution time. The
profiling path synchronizes only at explicitly measured phase boundaries and stays off in
normal training.

## Measurement protocol

- Use the pinned `.venv` environment and `quickstart/scripts/t2i/config_mps.yaml`.
- Warm up at least one optimizer step before recording timings.
- Record at least three optimizer steps, and include one checkpoint when measuring checkpoint
  cost.
- Report the median and individual samples, not only a single average.
- Record peak/current MPS allocation and whether `PYTORCH_ENABLE_MPS_FALLBACK` was enabled.
- Compare like with like: same cached batch, resolution, dtype, seed, accumulation, and model
  revision.
- Run unit tests after each slice. Run the real CPU-to-MPS forward parity and MPS learning gates
  before accepting a numerical or model-execution change.

## Step 1: Instrument and establish the baseline

Add an opt-in training profiler that separates:

1. cached-data wait;
2. transformer forward and loss construction;
3. backward;
4. gradient clipping plus optimizer/scheduler work;
5. host scalar readback/logging; and
6. checkpoint serialization.

The profiler must have zero device synchronization when disabled. It should produce structured,
JSON-serializable output suitable for comparing runs. Use the phase breakdown to confirm the next
four steps are attacking measured costs rather than assumed ones.

**Exit gate:** disabled-path unit tests pass; an MPS smoke run reports phase samples after a warmup
step without changing loss or checkpoint contents.

## Step 2: Benchmark gradient checkpointing off on MPS

The existing parity harness identifies checkpoint recomputation as the MPS bottleneck and already
disables it for its MPS run. Compare `gradient_checkpointing: true` and `false` using Step 1's
profile, including current/driver MPS allocation.

If the 512x512 LoRA workload fits with safe headroom, make checkpointing disabled in the MPS recipe
while retaining the configurable fallback for smaller-memory Macs. Do not change the general CUDA
default.

**Exit gate:** meaningful median iteration-time improvement, no out-of-memory event, finite loss,
and unchanged CPU-to-MPS forward parity.

## Step 3: Make CogView4 training attention masks device-resident and reusable

The selected Diffusers training attention processor currently creates the same quadratic mixed
attention mask inside each of the transformer's 28 blocks. First move all masks to the target device
once per batch. Then introduce a CogKit-owned processor path that constructs the invariant mask once
and reuses it across blocks without changing masking semantics.

Treat packed and non-packed training separately. The packed path has Python loops and scalar
materializations that need their own profile and parity gate; do not enable packing by default as
part of the non-packed optimization.

**Exit gate:** attention-mask construction disappears as a repeated hotspot, forward/backward
improve in the phase profile, and CPU-to-MPS `noise_pred`/loss parity remains within the documented
tolerances.

## Step 4: Configure inference placement once and benchmark warm requests

`before_generation` currently applies direct placement or installs offload hooks on every request,
and always enables VAE slicing and tiling. Make placement/offload configuration idempotent and move
long-lived API setup to pipeline initialization. Benchmark fresh-process latency separately from
second-and-later warm requests.

On the 64 GB reference machine, compare VAE tiling/slicing enabled and disabled at 512x512. Keep
the memory-saving behavior available and default conservatively where hardware capacity is unknown.

**Exit gate:** no duplicate hook installation, improved warm-request latency, stable memory across
repeated requests, and all inference parity gates remain green.

## Step 5: Decouple cached training data from model objects and overlap input work

After precomputation, the dataset still owns trainer/model-bound encoding methods and reopens
safetensors for every sample. Split precomputation from a lightweight cache-only dataset so the
training loader can safely use macOS `spawn` workers, persistent workers, and prefetching without
pickling the 6B trainer.

Benchmark worker counts 0, 1, and 2; unified memory means more workers are not automatically better.
Once input work overlaps correctly, reduce per-step scalar materialization/logging and evaluate a
less frequent or asynchronous single-device checkpoint path.

**Exit gate:** data-wait time is reduced without increased long-run memory pressure, cache misses
still fail or recompute safely, and samples remain byte-for-byte equivalent before device transfer.

## Acceptance record

For each step, append the commit, machine/OS, PyTorch revision, configuration, raw phase samples,
memory statistics, parity result, and keep/revert decision here. Do not replace baseline history with
only the winning number.

### Step 1 baseline — 2026-09-01

- Commit: `14c95eb`
- Hardware/runtime: M4 Max 64 GB, torch 2.12.1, native MPS available
- Workload: CogView4-6B LoRA, 512x512 bf16, batch size 1, gradient checkpointing enabled,
  `PYTORCH_ENABLE_MPS_FALLBACK=1`
- Profile window: one warmup step followed by four synchronized steps; step 5 included DCP save
- Losses remained finite: 1.29, 1.28, 1.12, 1.27, 1.31
- MPS allocation: 12.083 GB before training; 12.520 GB current / 15.086 GB driver after epoch

| Phase                | Samples (seconds)                      |     Median |
| -------------------- | -------------------------------------- | ---------: |
| Forward              | 3.254, 3.410, 3.458, 2.994             |    3.332 s |
| Backward             | 6.482, 6.271, 5.986, 5.853             |    6.128 s |
| Optimizer            | 0.331, 0.352, 0.354, 0.329             |    0.341 s |
| Scalar readback      | 0.000426, 0.000442, 0.000403, 0.000451 | 0.000434 s |
| Data wait            | 0.0082, 0.0100, 0.0122, 0.0104         |   0.0102 s |
| Whole training batch | 10.068, 10.034, 9.799, 9.177           |    9.916 s |
| DCP checkpoint       | 0.594                                  |    0.594 s |

Decision: keep the profiler. Backward is 62% of the median profiled batch and is the first target;
data loading and scalar readback are not material at this baseline. Proceed to the Step 2
checkpointing-on/off comparison before changing attention or input code.

### Step 2 accepted — gradient checkpointing off on MPS — 2026-09-03

- Commit: `212d3e4` (branch `perf/mps-step2-step3`)
- Hardware/runtime: M4 Max 64 GB, macOS 26.5.2, torch 2.12.1, `PYTORCH_ENABLE_MPS_FALLBACK=1`
- Workload: CogView4-6B LoRA, 512x512 bf16, batch size 1, `strategy: SINGLE`
- Full 2x2 matrix, raw samples, and method: `docs/benchmarks/APPLE_MPS_TRAINING_STEP_2026-09-03.md`
  plus `docs/benchmarks/apple_mps_training_step_2026-09-03.json`

| Phase           | checkpointing on | checkpointing off |      Delta |
| --------------- | ---------------: | ----------------: | ---------: |
| Forward         |          3.624 s |           3.784 s |      +4.4% |
| Backward        |          6.466 s |           2.945 s |     -54.5% |
| Optimizer       |          0.357 s |           0.350 s |      -2.0% |
| **Median step** |     **10.456 s** |       **7.098 s** | **-32.1%** |
| MPS reserved    |        15.086 GB |         25.154 GB |   +10.1 GB |

Loss sequence identical across every run (1.29, 1.28, 1.12, 1.27, 1.31), matching the Step 1
baseline. No out-of-memory event. Backward falling below forward is the expected LoRA shape once
recompute is gone: frozen base weights mean backward computes activation gradients only.

Decision: **keep.** `gradient_checkpointing: false` is now set in `quickstart/scripts/t2i/config_mps.yaml`
with the memory cost documented beside it. The `BaseArgs` default stays `True`, so CUDA and
smaller-memory Macs are unaffected and the fallback is one line of config.

### Step 3 accepted — CogKit-owned attention mask, built once — 2026-09-03

- Commit: `212d3e4`, `src/cogkit/finetune/diffusion/models/cogview/cogview4/attention.py`
- Same matrix, hardware, and workload as Step 2

`CogKitCogView4TrainingAttnProcessor` moves every mask to the target device once per forward and
reuses one mask across all 28 blocks instead of rebuilding it per block, dropping the mask
entirely when every token is valid. Packed training (`batch_flag`) is delegated to the upstream
processor untouched, as the plan requires.

Measured effect, holding gradient checkpointing fixed:

| Comparison                                  | upstream |   CogKit |       Delta |
| ------------------------------------------- | -------: | -------: | ----------: |
| median step, checkpointing on (run A vs C)  | 10.456 s | 10.319 s |       -1.3% |
| median step, checkpointing on (A vs C2)     | 10.456 s | 10.151 s |       -2.9% |
| median step, checkpointing off (B vs D)     |  7.098 s |  7.066 s |       -0.5% |
| two identical CogKit `gc_on` runs (C vs C2) | 10.319 s | 10.151 s | 1.6% spread |

**The effect does not clear the run-to-run spread.** A direct microbenchmark of the removed work
shows why: one mask rebuild costs 0.90 ms at 512x512 (2.01 ms at 1024x1024), so all 28 rebuilds
are ~25 ms against a 3.7 s forward -- 0.35% of a step.

**The plan's Step 3 premise was wrong.** Per-block mask construction is not a hotspot at any
resolution this lane trains at. No further mask work is warranted.

Decision: **keep, but not as a performance result.** Retained because it is provably equivalent
(`tests/test_cogview4_attention.py` checks outputs _and_ gradients against the upstream processor
across five mask shapes; the five real training losses are identical), because it removes 27
per-forward host-to-device mask copies that are also synchronization points, and because it is
what makes the mask-drop path possible. It is credited with no step-time improvement.

### Supporting measurement — where the training step actually goes

Recorded while accepting Steps 2 and 3, because it redirects Steps 4 and 5.

`aten/src/ATen/native/transformers/attention.cpp` routes MPS to the fused
`_scaled_dot_product_attention_math_for_mps` **only when grad is off**. With autograd recording,
MPS training always falls back to the composite `_scaled_dot_product_attention` math path, so
none of the fused MPS attention kernels ever run during training.

At one CogView4-6B attention block (B=1, H=32, S=1136, D=128, bf16) this is worth measuring
rather than assuming; see `Attention.mm` for the kernels that are being skipped.

Scale the per-block numbers by 28 layers and compare against the measured 3.75 s forward and
2.95 s backward: attention is a minority of the step, and the dense-mask and grad-fallback
penalties are each single-digit percentages of it. **The remaining time is in the transformer's
linear layers**, which is where a further training-side win has to come from -- not from masks,
and not from the data loader (data wait is 0.015 s, 0.2% of a step).

### Step 4 accepted — inference placement and VAE decode — 2026-09-03

- Branch `perf/mps-step4-inference` (parent `288d7a3`)
- Hardware/runtime: M4 Max 64 GB, macOS 26.5.2, torch `2.15.0a0+gitf6df965`, Python 3.14.2
- Workload: `THUDM/CogVideoX-2b`, 480x720, 9 frames, 1 step, bf16, `PYTORCH_ENABLE_MPS_FALLBACK=0`
- Full matrix, method, and the isolated decode microbenchmark:
  `docs/benchmarks/APPLE_MPS_VAE_DECODE_2026-09-03.md` plus
  `docs/benchmarks/apple_mps_vae_decode_2026-09-03.json`

**Half of this step was already done.** The plan says `before_generation` "applies direct
placement or installs offload hooks on every request". It does not: the `_COGKIT_LOAD_TYPE_ATTR`
guard landed in `7ae59d7` and short-circuits repeat calls. Diffusers' `enable_model_cpu_offload`
and `enable_sequential_cpu_offload` both call `remove_all_hooks()` internally, so there is no
hook stacking to fix either. Verified, not assumed. What was left: the API paid placement on its
first request, which now happens at service startup.

VAE slicing and tiling were unconditional. They are now `vae_slicing`/`vae_tiling` on
`before_generation` and a `vae_memory_saving` API setting, defaulting on:

| Load mode                | Warm latency        | VAE decode          | Driver max            |
| ------------------------ | ------------------- | ------------------- | --------------------- |
| `mps`                    | 23.56 s -> 19.56 s  | 16.78 s -> 12.91 s  | 24.24 GB -> 45.34 GB  |
| `sequential_cpu_offload` | 29.55 s -> 24.96 s  | 19.76 s -> 15.40 s  |  2.18 GB -> 10.13 GB  |

Decision: **keep the default on.** -17% warm latency is real, but 45.34 GB of driver allocation
on a 64 GB machine leaves under 19 GB of headroom, and the library cannot know the caller's
memory budget. The speed is available to anyone who states that they have the room.

Two things worth carrying forward. **Slicing does nothing at batch size 1** — it splits the
decode along the batch dimension — so tiling is the entire effect. And a first attempt at this
matrix produced identical numbers for both arms because `generate_video` forwards only
`load_type`, re-entering `before_generation` with the VAE flags unset; treating unset as "enable"
silently re-enabled tiling on every request. The harness now records the VAE flags read back
*after* the requests, so a run that did not measure what it claims is visibly invalid.

**Step 5's premise is retired.** The Step 1 profile measures data wait at 0.0092-0.015 s, 0.2% of
a training step. Decoupling the dataset from model objects may still be worth doing for `spawn`
correctness, but not as a performance step.
