# Apple MPS inference benchmark — 2026-09-01

This record compares CogKit inference under the pinned torch 2.12.1 environment and the
local PyTorch fork. Both runs use the cached `THUDM/CogView4-6B` weights, bfloat16, one
denoising step, identical CPU-generated inputs, and 512×512 output resolution.

**Headline: at n=5 no timing difference between the two runtimes survives run-to-run
noise.** Correctness is unaffected and fully reproducible. The single-invocation matrix
recorded earlier the same day (11:14) reported a "consistently faster" fork CPU path and a
24.5% MPS regression; repeated measurement does not support either claim. That section is
retained below, marked superseded, because the reason it was wrong is the useful part.

## Provenance

- Recorded: `2026-09-01T15:17:40-06:00` (supersedes `2026-09-01T11:14:28-06:00`)
- CogKit: `22593b117c65aba3c9dc81b2cbd229cbdedcde58`
- Host: MacBook Pro `Mac16,5`, Apple M4 Max, 16 CPU cores, 64 GB memory
- OS: macOS 26.5.2 (`25F84`)
- Baseline: torch `2.12.1` at `7269437d655783a26cba32aa88195b741ff496aa`,
  torchvision `0.27.1`
- Fork: torch `2.15.0a0+git9737e28` at
  `9737e2892a58d69b6cabdaedb33c4bd5809c54da`, torchvision `0.30.0a0+ac8d215`
- Shared stack: Diffusers `0.40.0.dev0`, Transformers `4.57.6`, Accelerate `1.14.0`

The fork build advanced from `cfca6b6` to `9737e28` between the two recordings; that range
includes the MPS copy-offset work (`aten/src/ATen/native/mps/operations/Copy.mm`) plus 30
upstream commits.

## Method

Five passes over a four-cell matrix (2 workloads × 2 runtimes), 20 measurements total.

- Fresh Python process per cell; no in-process warm-up iteration.
- Uniform 60 s cooldown before **every** cell, including the first.
- Runtime order alternates by pass — odd passes run baseline first, even passes run fork
  first — so cumulative thermal drift does not systematically favour one runtime.
- Model and system shader caches warm throughout.

Pass 1 followed other heavy work on the machine and is contaminated; it is included in the
headline figures and also excluded as a sensitivity check.

## Results

Median seconds across 5 passes, with observed range.

| Workload                                        | Device |          Baseline `2.12.1` |             Fork `9737e28` |
| ----------------------------------------------- | ------ | -------------------------: | -------------------------: |
| Synthetic embeddings → transformer → scheduler  | CPU    | 116.98 s [111.54 – 143.64] | 112.32 s [111.44 – 136.99] |
| Synthetic embeddings → transformer → scheduler  | MPS    |       6.14 s [5.57 – 6.54] |       6.10 s [5.23 – 9.37] |
| Real prompt → encoder → transformer → scheduler | CPU    | 117.58 s [116.56 – 128.00] | 117.76 s [116.93 – 130.13] |
| Real prompt → encoder → transformer → scheduler | MPS    |    25.70 s [21.14 – 28.43] |    24.75 s [19.57 – 26.37] |

## Paired comparison

Fork minus baseline within each pass, as a percentage; negative means the fork is faster.
Pairing within a pass controls for drift between passes.

| Cell          |                      per-pass Δ | median Δ | fork faster | excl. pass 1 |
| ------------- | ------------------------------: | -------: | ----------: | -----------: |
| synthetic CPU |   +10.5, −4.6, −0.1, −4.5, +0.5 |    −0.1% |         3/5 |        −2.3% |
| synthetic MPS |   +43.3, −6.1, +0.2, −5.3, +0.0 |    +0.0% |         2/5 |        −2.7% |
| real CPU      |    +3.6, −1.5, +0.2, −0.3, +0.5 |    +0.2% |         2/5 |        −0.1% |
| real MPS      | −7.2, −8.5, −22.8, +20.4, −11.2 |    −8.5% |         4/5 |        −9.8% |

**Every cell changes sign across passes.** Median effects are −0.1% to +0.2% on three of
four cells — indistinguishable from zero. The fourth, real-prompt MPS, has the largest
median (−8.5%) and the most suggestive count (4/5), but one pass had the fork 20.4%
_slower_; a range spanning −22.8% to +20.4% cannot support a performance claim.

Within-runtime variability is the reason. The baseline alone — identical software, nothing
touched between runs — moved 111.54 s to 143.64 s on the synthetic CPU cell (cv 9.8%) and
21.14 s to 28.43 s on real-prompt MPS (cv 11.4%). The noise floor is larger than any effect
being measured.

**Verdict: this harness cannot resolve a difference between the two runtimes.** Resolving
one requires a different design, not more repeats of this one — in-process warm-up
iterations, more denoising steps per measurement to amortize fixed overhead, and per-op
timing rather than whole-pipeline wall clock.

## Correctness

All 20 runs passed their CPU↔MPS parity gate. The metrics are deterministic — inputs are
CPU-generated and shared — so each runtime reproduces identical values on every pass:

| Runtime            | noise MRE | noise cosine | latent MRE | latent cosine |
| ------------------ | --------: | -----------: | ---------: | ------------: |
| baseline synthetic |  0.074905 |     0.997058 |   0.026034 |      0.999640 |
| fork synthetic     |  0.074056 |     0.997122 |   0.025729 |      0.999647 |
| baseline real      |  0.024561 |     0.999681 |   0.069126 |      0.998123 |
| fork real          |  0.024056 |     0.999694 |   0.067750 |      0.998203 |

The fork's values are **bit-identical to those recorded for `cfca6b6`** at 11:14. The MPS
copy-offset work between the two builds changed no numerical result — the expected and
desired outcome for a copy-path change. All metrics remain inside the documented acceptance
bounds.

## Commands

```bash
COGKIT_RUN_MPS_INFERENCE_PARITY=1 COGKIT_MPS_PARITY_SIZE=512 \
  <python> -m pytest -q -s \
  tests/test_inference_mps.py::test_cogview4_one_step_cpu_mps_latent_parity

COGKIT_RUN_MPS_INFERENCE_REAL_PROMPT_PARITY=1 COGKIT_MPS_COMPONENT_SIZE=512 \
  <python> -m pytest -q -s \
  tests/test_inference_mps.py::test_cogview4_real_prompt_cpu_mps_latent_parity
```

`<python>` is `.venv/bin/python` for the baseline and `.venv-pytorch-fork/bin/python` for
the fork. Per-run records are in `apple_mps_inference_2026-09-01.json` under `runs`.

## Superseded: single-invocation matrix (11:14)

The original recording took one measurement per cell and reported the fork's CPU path as
"consistently faster" (6.5–7.6%) with a 24.5% synthetic MPS regression.

| Workload    | Runtime        |      CPU |     MPS |
| ----------- | -------------- | -------: | ------: |
| Synthetic   | torch 2.12.1   | 134.66 s |  5.27 s |
| Synthetic   | fork `cfca6b6` | 124.44 s |  6.56 s |
| Real prompt | torch 2.12.1   | 139.35 s | 23.31 s |
| Real prompt | fork `cfca6b6` | 130.35 s | 20.86 s |

Re-measuring the **unchanged** baseline four hours later gave 120.43 s / 9.39 s on the
synthetic cell — a 78% swing on the MPS figure with no software change. Both original
conclusions sit inside that band. The n=1 design could not distinguish a runtime difference
from the machine's own variance, and the doc's own closing caveat ("repeat the runs before
making an optimization decision") was the correct instinct.
