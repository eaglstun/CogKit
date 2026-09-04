# Apple MPS VAE decode benchmark — 2026-09-03

`THUDM/CogVideoX-2b` text-to-video on an M4 Max 64 GB, measuring what VAE slicing and
tiling actually cost. 480x720, 9 frames, one denoising step, guidance 6.0, seed 42,
bfloat16, `PYTORCH_ENABLE_MPS_FALLBACK=0`.

**Recommendation: keep VAE memory saving on by default.** Turning it off buys ~17% warm
latency and costs 21 GB of MPS driver allocation — 45.34 GB on a 64 GB machine, leaving
under 19 GB for everything else. It is now a setting (`vae_memory_saving`), so a host with
known headroom can take the speed; the default stays conservative because the caller's
memory budget is not knowable from inside the library.

**VAE decode remains the dominant stage in every arm** — 66% to 79% of warm latency. The
2026-09-02 benchmark named it as the target and that still holds; tiling is simply not the
lever, because turning it off pays for its speed entirely in memory.

## Provenance

- Recorded: 2026-09-03, America/Boise
- CogKit: branch `perf/mps-step4-inference` (parent `288d7a3`)
- Host: MacBook Pro `Mac16,5`, Apple M4 Max, 16 CPU cores, 64 GB memory
- OS: macOS 26.5.2, arm64 · Python 3.14.2 (`.venv-pytorch-fork-314`)
- Torch: `2.15.0a0+gitf6df965` (local fork), Diffusers `0.40.0.dev0`,
  Transformers `4.57.6`, Accelerate `1.14.0`
- Raw result: `apple_mps_vae_decode_2026-09-03.json`

**These numbers are not comparable to `COGVIDEO_MPS_LOAD_MODES_2026-09-02.md`**, which ran
on torch 2.12.1 / Python 3.12. That run recorded a warm transformer stage of 14.50 s; the
same stage here is 5.74-6.97 s. Tiling cannot affect the transformer, so the difference is
the torch build. Both arms below were therefore re-measured on the same day and stack.

## Results — cold + 3 warm requests per arm, fresh subprocess each

| Load mode                | VAE saving | Cold    | Warm median | CV    | Transformer | VAE decode | Driver max |
| ------------------------ | ---------- | ------: | ----------: | ----: | ----------: | ---------: | ---------: |
| `mps`                    | on         | 23.79 s |     23.56 s | 3.72% |      6.48 s |    16.78 s |   24.24 GB |
| `mps`                    | **off**    | 20.19 s | **19.56 s** | 4.20% |      5.74 s | **12.91 s**|   45.34 GB |
| `sequential_cpu_offload` | on         | 30.64 s |     29.55 s | 3.54% |      6.53 s |    19.76 s | **2.18 GB**|
| `sequential_cpu_offload` | **off**    | 21.74 s |     24.96 s | 5.91% |      6.97 s |    15.40 s |   10.13 GB |

Deltas (off relative to on):

| Load mode                | Warm latency | VAE decode | Driver allocation |
| ------------------------ | -----------: | ---------: | ----------------: |
| `mps`                    |       -16.9% |     -23.1% |    +87.0% (+21 GB)|
| `sequential_cpu_offload` |       -15.5% |     -22.1% |  +365.8% (+8 GB)  |

Run-to-run spread here (CV 3.5-5.9%) is wider than the 2026-09-02 record's 0.4-1.3%;
another GPU-using application was running. The measured effect (-17%, -23%) is several
times that spread, so the direction and rough size are safe, but do not quote these to
three significant figures.

## Isolated decode microbenchmark

The pipeline harness samples memory at synchronized *stage boundaries*, so a transient
allocation that is freed before the boundary is invisible to it. Decoding a
`(1, 16, 3, 60, 90)` latent directly, with the VAE alone resident, shows what the pipeline
numbers hide:

| tiling | slicing | decode  | driver growth |
| ------ | ------- | ------: | ------------: |
| on     | on      | 14.68 s |     11.05 GB  |
| on     | off     | 14.81 s |      5.02 GB  |
| off    | on      | 11.95 s |     30.80 GB  |
| off    | off     | 11.88 s |     30.80 GB  |

Tiling is the whole effect: ~2.8 s and ~26 GB. **Slicing does nothing at this workload** —
it splits the decode along the batch dimension and the batch is 1. The two slicing rows
under `tiling=on` differ by 6 GB, which is allocator state between sequential runs rather
than a real slicing effect; do not read it as one.

Tiling engages here because the 60x90 latent exceeds this VAE's `tile_latent_min` of 30x45
(from `sample_height/width` 480/720). At a small enough resolution the flag is inert.

## Method note — the setting has to be verified, not requested

The first attempt at this matrix measured **the same numbers for both arms** (+0.1% decode,
0.0 GB). The cause was in the code under test: `generate_video` forwards only `load_type`,
so every request re-entered `before_generation` with the VAE flags unset, and unset was
being treated as "enable". All three warm requests of the "off" arm ran with tiling on.

Two changes came out of that. `None` now means *leave the current setting alone*, with the
conservative default applied once where placement is configured
(`tests/test_inference_mps.py::test_unset_vae_flags_do_not_reset_an_explicit_setting` is the
regression). And the harness now records `use_slicing`/`use_tiling` read off the VAE
**after** the requests, into `vae_flags_observed_after_requests`. A run whose observed flags
disagree with its config is invalid; all four arms above were checked.

## Commands

```bash
.venv-pytorch-fork-314/bin/python tools/benchmark_cogvideo_mps.py \
  --load-types mps,sequential_cpu_offload --vae-memory-saving off \
  --warm-repeats 3 --cooldown-seconds 30 --timeout 1800 --output vae_off.json
# then the same with --vae-memory-saving on
```
