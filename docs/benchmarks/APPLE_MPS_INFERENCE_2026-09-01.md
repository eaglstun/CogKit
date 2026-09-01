# Apple MPS inference benchmark — 2026-09-01

This record compares CogKit inference under the pinned torch 2.12.1 environment and the
local PyTorch fork. Both runs use the cached `THUDM/CogView4-6B` weights, bfloat16, one
denoising step, identical CPU-generated inputs, and 512×512 output resolution.

## Provenance

- Recorded: `2026-09-01T11:14:28-06:00`
- CogKit: `bd98a9b9bb74535d80f556a6b73b3a8a7ef1bec0`
- Host: MacBook Pro `Mac16,5`, Apple M4 Max, 16 CPU cores, 64 GB memory
- OS: macOS 26.5.2 (`25F84`)
- Baseline: torch `2.12.1` at `7269437d655783a26cba32aa88195b741ff496aa`,
  torchvision `0.27.1`
- Fork: torch `2.15.0a0+gitcfca6b6` at
  `cfca6b65486c97a2388977af003fe6446e123088`, torchvision
  `0.30.0a0+ac8d215`
- Shared stack: Diffusers `0.40.0.dev0`, Transformers `4.57.6`, Accelerate `1.14.0`

## Results

| Workload | Runtime | CPU | MPS | CPU/MPS | Fork vs baseline CPU | Fork vs baseline MPS |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Synthetic embeddings → transformer → scheduler | torch 2.12.1 | 134.66 s | 5.27 s | 25.55× | — | — |
| Synthetic embeddings → transformer → scheduler | fork `cfca6b6` | 124.44 s | 6.56 s | 18.97× | 7.6% faster | 24.5% slower |
| Real prompt → encoder → transformer → scheduler | torch 2.12.1 | 139.35 s | 23.31 s | 5.98× | — | — |
| Real prompt → encoder → transformer → scheduler | fork `cfca6b6` | 130.35 s | 20.86 s | 6.25× | 6.5% faster | 10.5% faster |

The fork's CPU path is consistently faster in these two samples. Its MPS result is
workload-dependent: the synthetic denoiser path regresses, while the composed real-prompt
path improves. These are single measured invocations in fresh Python processes, with warm
model and system shader caches but no explicit in-process warm-up iteration. Treat the
percentages as indicative; repeat the runs before making an optimization decision.

Every benchmark also passed its CPU↔MPS correctness gate. The metric changes between the
two torch builds are small and remain well inside the documented acceptance bounds.

## Commands

```bash
COGKIT_RUN_MPS_INFERENCE_PARITY=1 COGKIT_MPS_PARITY_SIZE=512 \
  <python> -m pytest -q -s \
  tests/test_inference_mps.py::test_cogview4_one_step_cpu_mps_latent_parity

COGKIT_RUN_MPS_INFERENCE_REAL_PROMPT_PARITY=1 COGKIT_MPS_COMPONENT_SIZE=512 \
  <python> -m pytest -q -s \
  tests/test_inference_mps.py::test_cogview4_real_prompt_cpu_mps_latent_parity
```

`<python>` was `.venv/bin/python` for the baseline and
`.venv-pytorch-fork/bin/python` for the fork.

## Raw records

```text
baseline synthetic:
torch=2.12.1 git=7269437d655783a26cba32aa88195b741ff496aa size=512 cpu=134.66s mps=5.27s noise_mean_relative_error=0.074905 latent_mean_relative_error=0.026034 noise_normalized_rmse=0.076724 latent_normalized_rmse=0.026835 noise_cosine_similarity=0.997058 latent_cosine_similarity=0.999640
1 passed in 144.13s

fork synthetic:
torch=2.15.0a0+gitcfca6b6 git=cfca6b65486c97a2388977af003fe6446e123088 size=512 cpu=124.44s mps=6.56s noise_mean_relative_error=0.074056 latent_mean_relative_error=0.025729 noise_normalized_rmse=0.075892 latent_normalized_rmse=0.026562 noise_cosine_similarity=0.997122 latent_cosine_similarity=0.999647
1 passed in 136.02s

baseline real prompt:
torch=2.12.1 git=7269437d655783a26cba32aa88195b741ff496aa text_length=64 size=512 cpu=139.35s mps=23.31s noise_mre=0.024561 noise_nrmse=0.025268 noise_cosine=0.999681 latent_mre=0.069126 latent_nrmse=0.061329 latent_cosine=0.998123
1 passed in 168.18s

fork real prompt:
torch=2.15.0a0+gitcfca6b6 git=cfca6b65486c97a2388977af003fe6446e123088 text_length=64 size=512 cpu=130.35s mps=20.86s noise_mre=0.024056 noise_nrmse=0.024732 noise_cosine=0.999694 latent_mre=0.067750 latent_nrmse=0.060071 latent_cosine=0.998203
1 passed in 157.17s
```
