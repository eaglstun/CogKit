# CogKit Apple Silicon Performance Plan

**Branch:** `perf/apple-silicon-training`  
**Baseline:** CogView4-6B LoRA, 512x512, bf16, batch size 1, approximately 7 s/it on
the documented M4 Max 64 GB run.  
**Rule:** a completed MPS run is not a correctness result. Every numerical change must
retain the CPU-to-MPS parity gates in `tests/test_mps_cpu_parity.py`.

**Current status:** Step 1 instrumentation is implemented on this branch; the real MPS baseline
run is pending. Enable it by setting `profile_steps: 3` (or more) in `config_mps.yaml`. Keep at
least one warmup step, and extend the window through a checkpoint when measuring serialization.

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
attention mask inside each of the transformer's 30 blocks. First move all masks to the target device
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
