# Apple MPS step-cost benchmark — 2026-09-02

Marginal MPS cost per denoising step for `THUDM/CogView4-6B`, bfloat16, 512×512, measured
with `test_cogview4_mps_step_cost_benchmark`. This harness exists because the whole-pipeline
timings in `APPLE_MPS_INFERENCE_2026-09-01.md` could not resolve a difference between the
two runtimes: their run-to-run spread (cv 5–22%) exceeded the effect being measured.

**Headline: the fork is 9.5% faster per denoising step, and the result is now separable.**
Baseline and fork ranges do not overlap at any step count under either load mode.

## Provenance

- Recorded: `2026-09-02T15:38:35-06:00`
- CogKit: `b7a1356868fc0c92eb37316777d7431d16bf2da6`, plus the uncommitted
  `tests/test_inference_mps.py` addition that provides this benchmark
- Host: MacBook Pro `Mac16,5`, Apple M4 Max, 16 CPU cores, 64 GB memory
- OS: macOS 26.5.2 (`25F84`)
- Baseline: torch `2.12.1` at `7269437d655783a26cba32aa88195b741ff496aa`,
  torchvision `0.27.1`
- Fork: torch `2.15.0a0+git9737e28` at
  `9737e2892a58d69b6cabdaedb33c4bd5809c54da`, torchvision `0.30.0a0+ac8d215`
- Shared stack: Diffusers `0.40.0.dev0`, Transformers `4.57.6`, Accelerate `1.14.0`

## Method

Three changes from the whole-pipeline harness, each targeting a specific defect:

1. **In-process warm-up, discarded.** The first iteration pays Metal shader compilation —
   measured at 1.88 s/it against 0.22 s once warm at 64². The old n=1 design capitalised
   that cost into every measurement.
2. **Step-count sweep with a least-squares fit.** Wall time is measured at 1, 4, and 8
   steps and regressed against step count. The intercept is fixed pipeline overhead; the
   slope, `marginal_per_step`, is the compute that scales with work done. A single
   whole-pipeline timing cannot separate the two, which is why it could not detect a
   kernel-level change.
3. **Explicit `torch.mps.synchronize()` before stopping the timer.** MPS dispatch is
   asynchronous. The parity tests satisfy this only incidentally — a trailing `.cpu()` sits
   inside their timed region — so a harmless-looking refactor there would silently reduce
   them to timing queue submission.

Five repeats per step count. CPU is excluded: at ~117 s for one step an 8-step CPU
measurement takes 15 minutes, and CPU serves as the parity oracle, not a performance target.

## Results — device-resident (`load_type=mps`)

| Metric              |          Baseline `2.12.1` |             Fork `9737e28` |         Δ | ranges    |
| ------------------- | -------------------------: | -------------------------: | --------: | --------- |
| 1 step (median)     |    2.375 s [2.342 – 2.429] |    2.166 s [2.123 – 2.198] |     −8.8% | separated |
| 4 steps (median)    |    9.588 s [9.447 – 9.679] |    8.675 s [8.614 – 8.753] |     −9.5% | separated |
| 8 steps (median)    | 19.562 s [19.300 – 20.005] | 17.721 s [17.577 – 18.033] |     −9.4% | separated |
| **marginal s/step** |                **2.457 s** |                **2.224 s** | **−9.5%** |           |
| fixed overhead      |                   −0.141 s |                   −0.118 s |           |           |

## Results — `load_type=cpu_model_offload`

| Metric              |          Baseline `2.12.1` |             Fork `9737e28` |          Δ | ranges    |
| ------------------- | -------------------------: | -------------------------: | ---------: | --------- |
| 1 step (median)     |    3.440 s [3.410 – 3.551] |    3.261 s [3.256 – 3.285] |      −5.2% | separated |
| 4 steps (median)    | 10.312 s [10.217 – 10.446] |    9.293 s [9.240 – 9.553] |      −9.9% | separated |
| 8 steps (median)    | 19.674 s [19.564 – 19.700] | 17.616 s [17.513 – 17.640] |     −10.5% | separated |
| **marginal s/step** |                **2.320 s** |                **2.052 s** | **−11.6%** |           |
| fixed overhead      |                    1.088 s |                    1.163 s |            |           |

## What the old harness was actually measuring

`cpu_model_offload` adds **+1.229 s of fixed overhead per call** over device-resident
execution — **32% of a one-step offloaded call**. The parity tests time exactly one step
under exactly that mode, so roughly a third of every number in the 2026-09-01 matrix was
submodule offload traffic rather than kernel time, sitting on top of unabsorbed shader
compilation.

Measurement spread collapsed accordingly. Whole-pipeline cv ran 5–22%; here the widest
range is ±1.8% (baseline, 8 steps) and every baseline/fork pair is fully separated.

Note that marginal per-step cost is slightly _lower_ under offload than device-resident for
both runtimes (−5.6% baseline, −7.7% fork). Both show it, so it is unlikely to be noise;
holding the full 6B model resident plausibly raises memory pressure. This was not
investigated.

## Attribution — read this before citing the 9.5%

The comparison is **torch 2.12.1 versus the fork at `9737e28`**, which differ by a major
version plus the fork's own patches. The 9.5% cannot be attributed to the MPS copy-offset
work specifically; no build between them was measured with this harness. Establishing that
requires timing intermediate commits — `cfca6b6` in particular, whose numerics are known
identical to `9737e28`.

## Commands

```bash
COGKIT_RUN_MPS_INFERENCE_BENCHMARK=1 COGKIT_MPS_BENCH_SIZE=512 \
  COGKIT_MPS_BENCH_STEPS=1,4,8 COGKIT_MPS_BENCH_WARMUP=1 COGKIT_MPS_BENCH_REPEATS=5 \
  COGKIT_MPS_BENCH_LOAD_TYPE=mps \
  <python> -m pytest -q -s \
  tests/test_inference_mps.py::test_cogview4_mps_step_cost_benchmark
```

`<python>` is `.venv/bin/python` for the baseline and `.venv-pytorch-fork/bin/python` for
the fork. Per-run records are in `apple_mps_step_cost_2026-09-02.json`.

## Tooling notes

- `ProfilerActivity.MPS` does not exist in either build — kineto has no MPS backend — so
  programmatic per-op GPU timing is unavailable. Per-op attribution needs
  `torch.mps.profiler.metal_capture` and Instruments.
- `torch.mps.Event(enable_timing=True)` is not dependable here. It reported 323 ms for a
  loop whose identical warm repeat took 151 ms by wall clock, then hung on `elapsed_time()`
  under the fork. Wall clock plus `torch.mps.synchronize()` is used instead. There is also
  a cosmetic `AttributeError` from `torch/mps/event.py:20` during interpreter shutdown.
