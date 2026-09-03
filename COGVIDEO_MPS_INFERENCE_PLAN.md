# CogVideo Inference on Apple Silicon (MPS)

**Status:** in progress — Steps 1–2 complete; Step 3 T2V component parity green 2026-09-02
**Branch:** `perf/apple-silicon-training`
**Primary target:** `THUDM/CogVideoX-2b`, text-to-video, bfloat16
**Second target:** `THUDM/CogVideoX-5b-I2V`, image-to-video, bfloat16

## Execution log

### 2026-09-02 — Step 1 and first real T2V run

- Added CLI `--num_frames` and `--guidance_scale` controls without changing the existing
  per-task defaults or passing the video-only option into image generation.
- Made scheduler and placement preparation idempotent for repeated requests.
- Added 14 passing hardware-free tests covering T2V/I2V routing, version-specific frame
  contracts, CPU-seeded generation, repeated MPS preparation, and CLI forwarding.
- Filled the local `THUDM/CogVideoX-2b` cache and ran the canonical 480×720 × 9-frame,
  one-step bfloat16 workload with model CPU offload.
- The first run localized a real MPS blocker before transformer execution:
  `transformer.patch_embed.pos_embedding` was a registered float64 buffer with shape
  `(1, 17776, 1920)`. Diffusers constructs it on CPU, but MPS cannot represent float64.
  CogKit now converts registered float64 buffers to float32 only when the target device is
  MPS; model parameters and CPU/CUDA paths are unchanged.
- After the fix, the tensor-video smoke passed both with fallback enabled (127.71 s total;
  14.36 s denoising) and with `PYTORCH_ENABLE_MPS_FALLBACK=0` (98.70 s total; 10.90 s
  denoising). Output was finite with shape `(1, 9, 3, 480, 720)` at 8 fps.
- The real `cogkit inference` path also passed with fallback disabled. Its temporary H.264
  MP4 was 720×480, contained exactly 9 frames at 8 fps, and reported a 1.125 s duration.
- These are bring-up timings, not benchmark claims: each is a single cold process and they
  include model loading, VAE decode, and test overhead.

### 2026-09-02 — Step 3 T2V CPU/MPS parity

- Added opt-in, real-model CPU-oracle tests for T5 prompt encoding, the transformer plus
  scheduler, and isolated temporal VAE decode. All MPS runs set
  `PYTORCH_ENABLE_MPS_FALLBACK=0`.
- The canonical 480×720 × 9-frame transformer/scheduler fixture passed in bfloat16: CPU
  244.43 s versus MPS 8.72 s, with mean relative error 0.001100, normalized RMSE 0.002275,
  and cosine similarity 0.999997 at the noise prediction, scheduler latent, and final output.
- A reduced 64×96 × 9-frame diagnostic localized the error growth: patch embedding cosine
  was 1.000000, block 00 was 0.999998, block 15 was 0.999897, block 29 was 0.999986, and
  final projection was 0.999994. The canonical regression limits are MRE and NRMSE below
  0.01 and cosine at least 0.9999.
- Full 226-token T5 encoding measured CPU 14.94 s versus MPS 1.43 s, scaled normalized RMSE
  0.048801, and cosine 0.998809. Its regression limits are scaled NRMSE below 0.055 and
  cosine at least 0.998.
- A 32×32 × 1-frame temporal VAE decode fixture measured CPU 8.55 s versus MPS 1.41 s,
  MRE 0.001929, normalized RMSE 0.003182, and cosine 0.999995. Its regression limits are
  MRE and NRMSE below 0.01 and cosine at least 0.9999.
- The first combined canonical component oracle was interrupted after 988.99 s while CPU
  was executing a temporal convolution. Splitting T5 and VAE made failures observable, and
  the micro VAE fixture supplies a practical per-change gate while the canonical composed
  T2V run still covers 480×720 decode on MPS.

Next: add dated machine-readable results and measure warm latency plus peak memory by load mode.

## Goal

Validate and ship CogVideo inference through PyTorch MPS without weakening numerical
correctness or hiding CPU fallbacks. The first supported surface is the existing Python API
and `cogkit inference` CLI. The Gradio inference UI follows once the same pipeline path is
green. A video HTTP endpoint is not part of this effort because CogKit's server currently
exposes only `/v1/images/generations`.

This is primarily a validation, controls, and measurement project. Shared MPS placement is
already present: `generate_video()` calls `before_generation()`, which supports direct MPS,
model CPU offload, and sequential CPU offload. CogVideo adds a much larger temporal working
set, a T5 text encoder, 3D transformer execution, temporal VAE encode/decode, and stricter
frame-shape constraints. Those are the parts that still need proof.

## Scope and order

1. **CogVideoX-2b T2V first.** It is the smallest production model and isolates generation
   from input-image VAE encoding. The minimal acceptance workload is 480×720, 9 frames
   (`8N+1`), one denoising step, one video, fixed seed.
2. **CogVideoX-5b-I2V second.** Its weights are already cached locally. It adds image
   preprocessing and VAE encoding after the T2V transformer/scheduler/decode path is known.
3. **CogVideoX1.5-5B last.** Its 768×1360 and `16N+1` contract makes it the expensive
   capacity test, not the bring-up fixture.

Direct full-pipeline MPS placement is experimental until measured. Model CPU offload is the
default candidate because unified memory does not make a 20+ GB resident pipeline free, and
the existing CogView measurements showed that full placement can make startup slower.

## Step 1 — Lock the public contract and cheap tests

Make the smallest workloads reachable and protect the already-working image path.

- Add `--num_frames` and `--guidance_scale` to the CLI and pass them only to the relevant
  generation function. Today the CLI cannot request a short video and silently uses the
  model's full default frame count.
- Make CogVideo scheduler/placement preparation idempotent so repeated server or Gradio
  requests do not rebuild the scheduler or stack offload hooks every time.
- Add fake-pipeline unit coverage for T2V and I2V routing, frame validation, scheduler setup,
  MPS placement, both offload modes, generator device behavior, and output shape checks.
- Keep `rand_generator()` CPU-backed for deterministic CPU/MPS input parity unless a measured
  bottleneck justifies changing it.
- Put CogVideo real-model tests in `tests/test_cogvideo_inference_mps.py`; do not enlarge the
  already expensive CogView parity module.

**Exit gate:** normal unit tests run without model weights or MPS hardware; CogView inference
tests remain green; the CLI can express the 9-frame/one-step smoke workload.

## Step 2 — Bring up one real T2V sample stage by stage

Use the pinned `.venv` first and record the exact Torch, Diffusers, Transformers, model
revision, macOS, and hardware versions.

Run these stages independently before asking the complete pipeline to emit an MP4:

1. load `CogVideoX-2b` in bfloat16;
2. encode a fixed positive and negative prompt with T5;
3. create CPU-seeded latents and rotary embeddings;
4. run one transformer denoise step on MPS;
5. run one scheduler step;
6. decode the 9-frame latent with VAE slicing and tiling;
7. postprocess frames and export an MP4.

First run with `PYTORCH_ENABLE_MPS_FALLBACK=1` to find functional blockers. Then run with
fallback disabled. Any operation that requires fallback must be named, timed, and recorded;
completion with an unknown CPU fallback is not an acceptance result. Prefer upstream
PyTorch/Diffusers behavior and small CogKit orchestration fixes—no custom Metal kernel or
vendored Diffusers pipeline in this phase.

**Exit gate:** the Python API and CLI each produce a readable 480×720, 9-frame video with
finite tensors, correct shape/range, deterministic seed behavior, and no unexplained device
transfer or fallback.

## Step 3 — Establish CPU-oracle numerical parity

A successful bfloat16 MPS run is not evidence of correctness. Reuse the CogView parity
discipline with identical CPU-generated inputs and stage-local captures.

Compare CPU and MPS at:

- T5 prompt and negative-prompt embeddings;
- first, middle, and last transformer blocks when localization is needed;
- final transformer noise prediction;
- scheduler output latent;
- isolated temporal VAE decode;
- final tensor video before PIL conversion.

For I2V, add input-image preprocessing and VAE encode parity. Report mean relative error,
normalized RMSE, cosine similarity, finite-value counts, and maximum absolute error. Set the
acceptance thresholds from the first clean measurements rather than copying CogView's bounds;
video attention and temporal VAE error propagation are different. Directional agreement is
mandatory even when magnitude is scale-sensitive.

Use a reduced spatial component fixture for quick localization, but retain one canonical
480×720 × 9-frame gate. CPU remains the oracle. Repeat the authoritative gate under the local
PyTorch fork only after it passes in the pinned environment.

**Exit gate:** T2V passes component and composed CPU↔MPS gates at the canonical workload,
with thresholds and measured values checked into a dated benchmark record.

## Step 4 — Bound memory and measure the useful modes

Video inference is likely memory-bound before it is compute-optimized. Measure instead of
assuming unified memory solves residency.

For direct MPS, model CPU offload, and sequential CPU offload, record:

- model-load, prompt-encode, denoise-step, VAE-decode, export, and total wall time;
- `torch.mps.current_allocated_memory()` and `driver_allocated_memory()` at every stage;
- process peak RSS/footprint using `/usr/bin/time -l` or `ru_maxrss`;
- cold first request and at least five warm in-process requests;
- whether fallback is enabled and which operations use it.

Warm the pipeline before comparing modes, synchronize MPS around timed regions, alternate
mode order, and cool down between heavy cells. The existing CogView benchmark showed that
single-invocation timings are too noisy for performance claims.

Start with VAE slicing and tiling enabled. Test one lever at a time: offload mode, attention
slicing if supported, decode tiling, then frame count. Do not combine changes before their
individual effect is known. Make preparation one-time so warm-request measurements do not
include scheduler reconstruction or repeated hook installation.

**Exit gate:** choose and document the recommended load mode with peak footprint and median
warm latency. The canonical workload must remain below the machine's practical memory limit
without swap-driven collapse or an OS kill.

## Step 5 — Expand to I2V, 1.5, UI, and documentation

- Run the same smoke, parity, and memory matrix on cached `CogVideoX-5b-I2V`, including VAE
  encode and image conditioning.
- Validate CogVideoX1.5-5B at 768×1360 with the minimum valid 17 frames before trying its
  default 81 frames.
- Route the verified modes through `gradio/gradio_infer_demo.py` and keep model preparation
  outside the per-click hot path.
- Update `README.md`, `docs/03-Inference/01-CLI.md`, and `docs/06-Apple-Silicon.md` with the
  exact supported models, commands, required environment variables, fallback caveats, memory
  expectations, and unsupported combinations.
- Add a dated result under `docs/benchmarks/` plus machine-readable JSON, following the
  CogView benchmark format.

**Exit gate:** T2V and I2V each have a real-model smoke test, a CPU-oracle parity gate, a
documented memory-safe recipe, and a user-facing CLI example. CogVideoX1.5 is labeled either
verified or explicitly unverified—never implied by the older-model result.

## Expected code touch points

- `src/cogkit/cli/inference.py` — expose video controls.
- `src/cogkit/python/generation/video.py` — validated video call contract and instrumentation.
- `src/cogkit/python/generation/util.py` — idempotent scheduler/device/offload preparation.
- `src/cogkit/utils/load.py` — only if model loading needs an explicit low-memory option.
- `gradio/gradio_infer_demo.py` — after the core path is verified.
- `tests/test_cogvideo_inference_mps.py` — cheap routing tests plus opt-in real-model gates.
- `docs/benchmarks/` and Apple Silicon docs — provenance, results, and supported recipes.

## Definition of done

- [x] `CogVideoX-2b` T2V produces a valid MP4 through Python and CLI on MPS.
- [ ] `CogVideoX-5b-I2V` produces a valid conditioned MP4 on MPS.
- [x] CPU↔MPS parity passes at text, transformer, scheduler, VAE, and composed-output stages.
- [ ] Unsupported/fallback operations are explicit; no success claim relies on an unknown CPU path.
- [ ] Recommended offload mode is backed by repeatable warm timing and peak-footprint data.
- [ ] Short-video CLI controls, unit tests, benchmark records, and Apple Silicon docs are landed.
- [ ] CogView inference and training parity tests remain green.
