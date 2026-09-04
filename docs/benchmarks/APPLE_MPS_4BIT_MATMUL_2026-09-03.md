# Native 4-bit matmul (bitsandbytes Metal) on the CogKit QLoRA lane — 2026-09-03

**Machine:** M4 Max, 64 GB, macOS 26.5 · **Env:** `.venv-pytorch-fork-314` (Python 3.14,
torch `2.15.0a0+gitf6df965`), bitsandbytes fork `0.50.3.dev0` built with
`-DCOMPUTE_BACKEND=mps`, `BNB_MPS_REQUIRE_NATIVE=1`
**Workload:** `config_mps_qlora.yaml` — CogView4-6B LoRA, NF4 base, 512×512, batch 1, 5 steps
(1 warmup + 4 profiled), `gradient_checkpointing: false`

**Headline: the native 4-bit matmul cuts the QLoRA step 15.3%, and cuts QLoRA's overhead over
plain bf16 LoRA from +36% to +14%.** Both directions of the matmul are now native for bf16 —
forward via `MPSGraph` (`MPSMatrixMultiplication` has no bf16), backward via a new fused
`gemm_4bit_backward` op that replaces a dequant-plus-torch-matmul composition.

## 1. Native vs fallback, three arms

Toggled **in place against one binary** with `BNB_MPS_DISABLE_BF16_GEMM` /
`BNB_MPS_DISABLE_BF16_GEMM_BWD`, so nothing but the routing differs. **n=6 per arm, half the reps
with the arm order reversed.**

| arm | forward | backward | step |
| --- | --- | --- | --- |
| neither (pre-native) | 4.784 ± 0.363 s | 4.340 ± 0.375 s | 9.394 ± 0.716 s |
| forward only | 4.130 ± 0.468 s | 4.171 ± 0.435 s | 8.575 ± 0.891 s |
| **both** | **4.055 ± 0.264 s** | **3.638 ± 0.179 s** | **7.960 ± 0.431 s** |
| both vs neither | **−15.2%** | **−16.2%** | **−15.3%** |

The fused backward alone accounts for −12.8% of backward and −7.2% of step.

**Built-in error bar.** `forward only` should not move the backward and reports −3.9%; `both`
should not move the forward relative to `forward only` and reports −1.8%. So the per-stage noise
floor is ~2–4%, and the effects above clear it by 3–6x. The winning arm is also the tightest in
the table (± 0.179 vs ± 0.375), which is what removing a per-layer cross-queue sync should do.

**Arm order matters.** Reversing it moved the same arm's step time by −0.50 s to +0.64 s — as
large as the effect being measured. Half the reps were run reversed for exactly this reason.

## 2. bf16 LoRA vs NF4 QLoRA, re-measured head to head

Both arms measured today on the quiet machine, n=3 each, alternated.

| | forward | backward | step | MPS allocated | MPS reserved |
| --- | --- | --- | --- | --- | --- |
| bf16 LoRA (`config_mps.yaml`) | 3.597 s | 3.016 s | 6.896 s | 12.52 GB | 24.22 GB |
| NF4 QLoRA, native matmul | 3.945 s | 3.614 s | 7.833 s | 3.72 GB | 16.18 GB |
| delta | +9.7% | +19.8% | **+13.6%** | **−70.3%** | −33.2% |

Against the *fallback* QLoRA arm from §1 (9.394 s), plain bf16 was +36% ahead. Native brings that
to +13.6%. **QLoRA now costs ~14% step time for 70% of live MPS allocation**, where it used to
cost ~36%.

## 3. Why `APPLE_MPS_QLORA_2026-09-03.md` needs reading with care

That record puts NF4 at **+14.1%** over bf16 — numerically almost identical to the +13.6% above,
which makes it look as if nothing changed. It is a coincidence, and the absolutes show why:

| | that record | today, quiet machine |
| --- | --- | --- |
| bf16 step | 7.67 s | 6.896 s |
| NF4 step | 8.75 s (fallback) | 9.394 s (fallback) / 7.833 s (native) |

Its bf16 arm was ~10% slow and its NF4 arm ~7% fast relative to today, and the two errors happened
to cancel in the ratio. bf16 LoRA does not call bitsandbytes at all, so nothing in this work could
have changed it — the difference is the machine, which spent part of that day under a load average
of 200–298. **The ratio in that record survived; its absolute numbers did not, and its +14.1%
should not be read as "native changed nothing".**

## 4. Component microbenchmarks

nf4 / blocksize 64, 30 iters / 8 warmup, 64 MB `clone()` control 0.298 → 0.287 ms across the table
(≈427 GB/s, stable — the table is internally comparable).

`gemm_4bit`, native vs dequant + `F.linear`, N=K=4096:

| M | bf16 | fp16 | fp32 |
| --- | --- | --- | --- |
| 8 | 2.09x | 2.16x | 1.22x |
| 64 | 1.94x | 1.94x | 1.04x |
| 512 | 1.09x | 1.80x | 1.24x |
| 2048 | 1.03x | 0.96x | 0.93x |

The win is the single command buffer, so it shrinks as the GEMM starts to dominate. At the
CogView4 training shapes (M=1024/1280 × the four `(N, K)`) the forward lands at 0.98–1.16x and the
**backward at 1.14–1.31x** — better, because the backward needs no transpose and has no bias
epilogue.

## 5. Reproduce

```bash
# component tables
BNB_MPS_REQUIRE_NATIVE=1 .venv-pytorch-fork-314/bin/python \
  ~/Documents/dev/bitsandbytes/benchmarks_wip/bench_gemm_baseline.py --cogkit

# end-to-end arms (NOTE: TORCHRUN must point at the 3.14 venv -- it is the only one with bnb)
cd quickstart/scripts/t2i
TORCHRUN=../../../.venv-pytorch-fork-314/bin/torchrun \
BNB_MPS_DISABLE_BF16_GEMM=0 BNB_MPS_DISABLE_BF16_GEMM_BWD=0 \
BNB_MPS_REQUIRE_NATIVE=1 COGKIT_CONFIG=config_mps_qlora.yaml bash start_train_mps.sh
```

Check `uptime` first and alternate the arm order. Raw stats: `apple_mps_4bit_matmul_2026-09-03.json`.
