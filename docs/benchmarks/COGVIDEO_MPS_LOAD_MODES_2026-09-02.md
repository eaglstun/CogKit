# CogVideo MPS load-mode benchmark — 2026-09-02

`THUDM/CogVideoX-2b` text-to-video inference on a 64 GB M4 Max, using bfloat16,
480×720 output, 9 frames, one denoising step, guidance scale 6.0, and seed 42.
`PYTORCH_ENABLE_MPS_FALLBACK=0` was set for every run.

**Recommendation:** use `sequential_cpu_offload` by default on a 64 GB Mac. It matched or
slightly beat direct MPS latency in both mode orders while holding observed MPS driver
allocation near 11 GB instead of 61 GB. Direct `mps` is an experimental speed option for a
quiet 64 GB machine, but its small latency advantage over the other modes is not reliable
enough to justify leaving less than 3 GB of nominal memory headroom.

## Provenance

- Recorded: 2026-09-02, America/Boise
- CogKit: `ae4cddfbbd07c3ff5b96dfbb69d57ceec4e3de67`, plus the benchmark harness
  being measured
- Host: MacBook Pro `Mac16,5`, Apple M4 Max, 16 CPU cores, 64 GB memory
- OS: macOS 26.5.2, arm64
- Python: 3.12.13
- Torch: 2.12.1, git `7269437d655783a26cba32aa88195b741ff496aa`
- Diffusers: 0.40.0.dev0
- Transformers: 4.57.6
- Accelerate: 1.14.0
- Raw result: `cogvideo_mps_load_modes_2026-09-02.json`

## Method

Each load mode ran in a fresh subprocess so Accelerate hooks, allocator caches, and process
peak RSS could not leak between modes. The main matrix used one cold request followed by five
warm requests, with a 30-second cooldown between modes. A reverse-order confirmation used
three warm requests each for sequential offload and direct MPS.

The harness synchronizes MPS around the whole request and around prompt encoding, transformer
execution, scheduler steps, VAE decode, and postprocessing. It records wall time, call counts,
`torch.mps.current_allocated_memory()`, `torch.mps.driver_allocated_memory()`, and process peak
RSS (`ru_maxrss`). Stage synchronization makes the breakdown deterministic but means these
numbers describe this diagnostic harness, not unsynchronized throughput in another runtime.

MPS allocation maxima are the largest values observed at synchronized stage boundaries. They
are not hardware-counter peaks. Driver allocation and process RSS measure different layers of
the unified-memory stack and must not be added together.

## Main matrix — five warm requests

| Load mode | Cold total | Warm median [min–max] | CV | Text | Transformer | VAE decode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `sequential_cpu_offload` | 72.23 s | **70.80 s** [70.49–71.22] | 0.36% | 1.18 s | 14.54 s | 54.98 s |
| `mps` | **70.39 s** | 72.18 s [70.57–73.07] | 1.27% | 1.52 s | **14.50 s** | 55.96 s |
| `cpu_model_offload` | 81.74 s | 80.57 s [74.52–82.79] | 4.31% | 2.33 s | 16.14 s | 58.41 s |

All stage columns are warm medians. Scheduler and postprocessing each took less than 7 ms;
VAE decode consumed roughly 73–78% of total warm latency and is the next optimization target.

## Memory

| Load mode | Observed live MPS max | Observed driver max | Process peak RSS |
| --- | ---: | ---: | ---: |
| `sequential_cpu_offload` | **0.22 GB** | **10.96 GB** | 17.96 GB |
| `cpu_model_offload` | 9.53 GB | 43.11 GB | 17.96 GB |
| `mps` | 13.54 GB | 61.37 GB | **17.54 GB** |

Sequential offload reduced observed driver allocation by 82% relative to direct MPS and 75%
relative to model offload. Similar process RSS across modes is expected because CPU-resident
weights remain part of the process while the MPS driver counter includes allocator-managed
Metal memory.

## Reverse-order confirmation

The main matrix ran model offload → direct MPS → sequential offload. A second run reversed the
two leading modes:

| Load mode | Warm repeats | Warm median [min–max] | Transformer | VAE decode |
| --- | ---: | ---: | ---: | ---: |
| `sequential_cpu_offload` | 3 | **82.56 s** [80.69–85.28] | 16.18 s | 64.89 s |
| `mps` | 3 | 83.86 s [82.12–84.43] | 17.24 s | 65.22 s |

Both modes shifted 11–12 seconds slower between runs, indicating meaningful thermal or system
state sensitivity. Sequential offload nevertheless retained a small 1.6% lead; it led by 1.9%
in the main matrix. The safe conclusion is latency parity within machine-state noise, paired
with a large and repeatable sequential-offload memory advantage.

## Command

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python tools/benchmark_cogvideo_mps.py \
  --load-types cpu_model_offload,mps,sequential_cpu_offload \
  --warm-repeats 5 --cooldown-seconds 30 --timeout 1200 \
  --output docs/benchmarks/cogvideo_mps_load_modes_2026-09-02.json
```

This is a minimum valid structural workload, not a 50-step production benchmark. Step 3's
CPU-oracle tests establish correctness; this benchmark intentionally measures performance and
memory using the already-validated path.
