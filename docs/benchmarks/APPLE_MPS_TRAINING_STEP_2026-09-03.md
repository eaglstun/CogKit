# Apple MPS training step benchmark — 2026-09-03

Synchronized phase profile for `THUDM/CogView4-6B` LoRA training, bfloat16, 512×512, batch
size 1, on the single-device MPS lane. Two changes are measured against each other in a
2×2: gradient checkpointing on/off (`APPLE_SILICON_PERFORMANCE_PLAN.md` Step 2) and the
upstream vs. CogKit-owned attention-mask construction (Step 3).

**Headline: turning gradient checkpointing off is worth 32% of the median step. Building the
attention mask once per forward instead of once per block is not measurable at this
workload — it is inside the run-to-run spread.**

## Provenance

- Recorded: 2026-09-03, America/Boise
- CogKit: `212d3e4`, branch `perf/mps-step2-step3`
- Host: MacBook Pro `Mac16,5`, Apple M4 Max, 16 CPU cores, 64 GB memory
- OS: macOS 26.5.2, arm64
- Python 3.12.13, torch 2.12.1, `PYTORCH_ENABLE_MPS_FALLBACK=1`, `COGKIT_DEVICE=mps`
- Workload: `quickstart/scripts/t2i/config_mps.yaml`, `strategy: SINGLE`, 5 training steps
- Profile window: one warmup step then four synchronized steps; step 5 carried the DCP save
- Raw result: `apple_mps_training_step_2026-09-03.json`

## Method

Each cell is a fresh `torchrun` process over the same cached batch, same seed, same five
prompts, run back to back with a 45-second cooldown. The `gc_on` cells were run first and
last so that a repeat of an identical configuration brackets the matrix; that repeat is the
run-to-run spread any claimed effect has to beat.

All five runs produced the identical loss sequence — 1.29, 1.28, 1.12, 1.27, 1.31 — matching
the 2026-09-01 baseline. Neither change moves the numbers; only their cost.

Only the phase timers synchronize MPS, and only inside the profile window
(`TrainingProfiler`), so these are execution times rather than command-submission times.

## Results — medians of four profiled steps

| Run | attention mask | grad ckpt | Forward | Backward | Optimizer | **Step** |
| --- | -------------- | --------- | ------: | -------: | --------: | -------: |
| A   | upstream       | on        | 3.624 s |  6.466 s |   0.357 s | 10.456 s |
| C   | CogKit         | on        | 3.666 s |  6.264 s |   0.349 s | 10.319 s |
| C2  | CogKit         | on        | 3.638 s |  6.184 s |   0.353 s | 10.151 s |
| B   | upstream       | off       | 3.784 s |  2.945 s |   0.350 s |  7.098 s |
| D   | CogKit         | off       | 3.751 s |  2.953 s |   0.356 s |  7.066 s |

Data wait stayed at 0.015 s and scalar readback under 0.001 s in every run; neither is
material at this workload.

### Step 2 — gradient checkpointing off

Holding the attention path fixed, removing recompute takes **backward from 6.47 s to 2.95 s
(−54%)** and the **median step from 10.46 s to 7.10 s (−32%)**.

Backward now costs _less_ than forward, which is what LoRA should look like: the base weights
are frozen, so backward computes activation gradients only, and there is no longer a full
forward recomputation stacked on top of it.

Forward rises slightly (3.62 s → 3.78 s) because it now writes activations it used to discard.

### Step 3 — mask built once per forward

Holding gradient checkpointing fixed, the CogKit processor measures −1.3% (A→C) and −2.9%
(A→C2) with checkpointing on, and −0.5% (B→D) with it off. The two identical `gc_on` CogKit
runs, nine minutes apart, differ by 1.6%. **The effect is not separable from machine-state
drift at this sample size.**

A direct microbenchmark of the construction it removes explains why (`B=1`, text 112 + image
1024 tokens, bf16, MPS):

| Workload  | sequence | one rebuild | 28 rebuilds = one forward |
| --------- | -------: | ----------: | ------------------------: |
| 512×512   |     1136 |     0.90 ms |                   25.1 ms |
| 1024×1024 |     4208 |     2.01 ms |                   56.3 ms |

25 ms against a 3.7 s forward is 0.7% of the forward and 0.35% of the step. The plan's
premise — that per-block mask construction is a repeated hotspot — is quantitatively wrong at
this resolution. It does not become one at 1024×1024 either.

## Memory

| Run                        | MPS allocated after epoch | MPS reserved after epoch |
| -------------------------- | ------------------------: | -----------------------: |
| gradient checkpointing on  |                  12.53 GB |             **15.09 GB** |
| gradient checkpointing off |                  12.53 GB |             **25.15 GB** |

Turning checkpointing off costs about **10 GB of MPS driver-reserved memory**. That is
comfortable on the 64 GB reference machine and is why `config_mps.yaml` now defaults it off,
but it is the reason the setting stays configurable: a 16 GB or 24 GB Mac should keep
recompute. The CUDA default in `BaseArgs` is unchanged.

## Decision

- **Step 2: keep.** `gradient_checkpointing: false` is now the MPS recipe default, with the
  memory cost documented in `config_mps.yaml` next to the setting.
- **Step 3: keep, but not as a performance result.** The CogKit processor is retained because
  it is provably equivalent (unit-tested against upstream including gradients, and identical
  real losses here), because it removes 27 per-forward host-to-device mask copies that are
  also synchronization points, and because it is what makes the mask-drop path possible. It is
  **not** credited with a step-time improvement, and no further mask work is warranted.
