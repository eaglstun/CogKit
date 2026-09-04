# Apple MPS QLoRA benchmark — 2026-09-03

CogView4-6B LoRA training on an M4 Max 64 GB, comparing the bf16 base transformer against
an NF4-quantized base (QLoRA, `low_vram: true`). Both arms ran on the same machine, the same
PyTorch build and the same venv on the same day, so the two columns are directly comparable
to each other — but **not** to the numbers in `APPLE_MPS_TRAINING_STEP_2026-09-03.md`, which
were taken on torch 2.12.1.

> [!NOTE]
> **Superseded in part by `APPLE_MPS_4BIT_MATMUL_2026-09-03.md`.** The NF4 arm below predates the
> native 4-bit matmul in the bitsandbytes Metal fork, so it measures the *fallback* configuration.
> The absolute step times here are also off: this run's bf16 arm reads ~10% slow and its NF4 arm
> ~7% fast against a later quiet-machine re-measurement, because part of that day the box sat at a
> load average of 200–298. The two errors happened to cancel, so the +14.1% ratio survived — which
> makes it easy to misread as "the native matmul changed nothing". It did not: measured
> consistently on a quiet machine, native cuts the QLoRA step 15.3% and cuts QLoRA's overhead over
> bf16 from +36% to +13.6%. **Take the ratio here, not the seconds.**

**Headline: NF4 costs 14% step time and returns 70% of live MPS allocation.** On a 64 GB Mac
that is a trade you take only when you need the memory; on a 16-24 GB Mac it is what makes
the workload fit at all.

## Provenance

- Recorded: 2026-09-03, America/Boise
- CogKit: branch `feat/mps-qlora`
- Host: MacBook Pro `Mac16,5`, Apple M4 Max, 16 CPU cores, 64 GB memory
- OS: macOS 26.5.2, arm64
- Python: 3.14.2 (`.venv-pytorch-fork-314`)
- Torch: `2.15.0a0+gitf6df965` (local fork build, `USE_DISTRIBUTED=1`)
- bitsandbytes: `0.50.3.dev0`, local fork, source build `-DCOMPUTE_BACKEND=mps`
  (native Metal kernels; `bnb_mps_check_buffer_contract` passes under this torch)
- Diffusers `0.40.0.dev0`, Transformers `4.57.6`
- Workload: 512x512, bf16 compute, batch size 1, `strategy: SINGLE`,
  `gradient_checkpointing: false`, `PYTORCH_ENABLE_MPS_FALLBACK=1`
- Configs: `quickstart/scripts/t2i/config_mps.yaml` vs `config_mps_qlora.yaml`
  (identical except `low_vram`)

## Results — 5 training steps, 1 warmup + 4 profiled

| Phase                | bf16     | QLoRA NF4 |      Delta |
| -------------------- | -------: | --------: | ---------: |
| Forward              |  3.777 s |   4.120 s |      +9.1% |
| Backward             |  3.463 s |   4.095 s |     +18.2% |
| Optimizer            |  0.295 s |   0.283 s |      -4.0% |
| Data wait            | 0.0099 s |  0.0092 s |          — |
| Checkpoint           |  0.614 s |   0.182 s |     -70.4% |
| **Median step**      | **7.67 s** | **8.75 s** | **+14.1%** |
| MPS allocated (live) | 12.521 GB |  3.721 GB | **-70.3%** |
| MPS reserved         | 24.076 GB | 17.053 GB | **-29.2%** |

The checkpoint difference is not a QLoRA speedup: `low_vram` saves a plain LoRA adapter
(224 MB) via `save_lora` instead of a DCP checkpoint of model + optimizer state.

Backward pays more than forward (+18.2% vs +9.1%), which is the expected shape: the LoRA
backward dequantizes the frozen NF4 base weights again to form input gradients, and MPS has
no fused 4-bit backward. Note also that `gemm_4bit` falls back to pure-torch for bf16 in this
bitsandbytes build (`docs/apple_silicon/MPS_STATUS.md` §2), so the general-M matmul is not
running the native Metal path here.

## Correctness

Per the standing rule, a completed MPS run is not a correctness result.

**Whole model, bf16 vs NF4 on MPS**, same cached batch with noise and timestep injected
(`test_qlora_matches_bf16_on_fixed_batch`):

| Quantity                    |    bf16 |      NF4 |    Delta |
| --------------------------- | ------: | -------: | -------: |
| loss                        | 1.210938 | 1.226562 |   +1.29% |
| `noise_pred` mean rel. diff |       — |          |   21.04% |
| `noise_pred` cosine         |       — |          | 0.979001 |

The 21% elementwise movement is what quantizing a 6B transformer to 4 bits does; it is not
device-specific and would look the same on CUDA. What matters for training is that the
optimizer sees essentially the same signal (loss within 1.3%) and the prediction still points
the same direction (cosine 0.979). QLoRA then trains adapters against the quantized base, so
the quantized model — not the bf16 one — is the thing being fitted.

**Device parity is gated at the layer, not the model** (`tests/test_qlora_mps.py`):

- NF4 packed codes and absmax are **bit-exact** between CPU and MPS.
- `Linear4bit` forward output and *input gradients* match the CPU reference to
  < 1e-3 mean relative difference, in fp32 and bf16, with and without double quantization.

There is deliberately no whole-model CPU NF4 oracle: the diffusers bitsandbytes quantizer
**rejects a `cpu` device_map outright** (verified — `validate_environment(device_map={"": "cpu"})`
raises), so a quantized model cannot be placed on CPU through the loading path CogKit uses.
`BaseTrainer._check_device_compat` now says this up front instead of letting the run die
later inside a Metal kernel with a device-mismatch error.

**Backward is gated separately** by `test_qlora_can_learn`: an overfit-one-batch run with the
batch, noise and timestep fixed must drive the loss monotonically down. A dequantize path
returning plausible garbage gradients would still complete a training run and still print a
believable loss, so this is the test that actually indicts it.

## A note on per-step loss

The Step 2 record states the bf16 loss sequence was "identical across every run"
(1.29, 1.28, 1.12, 1.27, 1.31) on torch 2.12.1. That does **not** reproduce on torch 2.15:
a bf16 rerun on 2026-09-03 gave 1.10, 1.34, 1.20, 1.30, 1.29 and the QLoRA run gave
1.12, 1.36, 5.56, 1.32, 1.32. `get_timestep` draws a random timestep per step and flow-matching
loss varies strongly with it, so single-step loss values are not a comparison instrument here
and the 5.56 is not evidence of a QLoRA fault. Use the fixed-batch gates above instead; why the
sequence was stable under the older torch was not investigated.

## Commands

```bash
# both arms, from quickstart/scripts/t2i
TORCHRUN=<repo>/.venv-pytorch-fork-314/bin/torchrun COGKIT_CONFIG=config_mps.yaml       bash start_train_mps.sh
TORCHRUN=<repo>/.venv-pytorch-fork-314/bin/torchrun COGKIT_CONFIG=config_mps_qlora.yaml bash start_train_mps.sh

# correctness gates
.venv-pytorch-fork-314/bin/python -m pytest tests/test_qlora_mps.py -q
.venv-pytorch-fork-314/bin/python -m pytest tests/test_mps_cpu_parity.py -k qlora -q -s
```
