#!/usr/bin/env python3
"""Benchmark CogVideo inference load modes on Apple Silicon in isolated processes."""

from __future__ import annotations

import argparse
import functools
import gc
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Sequence, cast

import accelerate
import diffusers
import torch
import transformers
from torch.utils.hooks import RemovableHandle

from cogkit.python import generate_video
from cogkit.python.generation.util import before_generation
from cogkit.types import LoadType
from cogkit.utils import load_pipeline

_STAGES = (
    ("text_encode", "pipeline", "encode_prompt"),
    ("scheduler", "scheduler", "step"),
    ("vae_decode", "vae", "decode"),
    ("postprocess", "video_processor", "postprocess_video"),
)
_ALL_STAGE_NAMES = ("text_encode", "transformer", "scheduler", "vae_decode", "postprocess")


def _sync_mps() -> None:
    torch.mps.synchronize()


def _memory_snapshot() -> dict[str, int]:
    return {
        "current_allocated_bytes": torch.mps.current_allocated_memory(),
        "driver_allocated_bytes": torch.mps.driver_allocated_memory(),
        "process_peak_rss_bytes": _peak_rss_bytes(),
    }


def _peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def _summarize(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    median = statistics.median(values)
    mean = statistics.fmean(values)
    standard_deviation = statistics.pstdev(values) if len(values) > 1 else 0.0
    return {
        "count": len(values),
        "median_seconds": median,
        "mean_seconds": mean,
        "min_seconds": min(values),
        "max_seconds": max(values),
        "standard_deviation_seconds": standard_deviation,
        "coefficient_of_variation": standard_deviation / mean if mean else 0.0,
    }


class StageTimer:
    """Synchronize and time selected pipeline stages without changing Diffusers."""

    def __init__(self, pipeline: Any) -> None:
        self.pipeline = pipeline
        self.totals = {stage: 0.0 for stage in _ALL_STAGE_NAMES}
        self.calls = {stage: 0 for stage in _ALL_STAGE_NAMES}
        self.memory_samples: list[dict[str, Any]] = []
        self.request = "unassigned"
        self._originals: list[tuple[Any, str, Callable[..., Any]]] = []
        self._module_hooks: list[RemovableHandle] = []
        self._transformer_started = 0.0

    def __enter__(self) -> StageTimer:
        for stage, owner_name, method_name in _STAGES:
            owner = self.pipeline if owner_name == "pipeline" else getattr(self.pipeline, owner_name)
            original = getattr(owner, method_name)
            self._originals.append((owner, method_name, original))
            setattr(owner, method_name, self._timed(stage, original))
        self._module_hooks.extend(
            (
                self.pipeline.transformer.register_forward_pre_hook(self._transformer_start),
                self.pipeline.transformer.register_forward_hook(self._transformer_end),
            )
        )
        return self

    def __exit__(self, *_: object) -> None:
        for owner, method_name, original in reversed(self._originals):
            setattr(owner, method_name, original)
        self._originals.clear()
        for handle in self._module_hooks:
            handle.remove()
        self._module_hooks.clear()

    def checkpoint(self) -> tuple[dict[str, float], dict[str, int]]:
        return self.totals.copy(), self.calls.copy()

    def delta(
        self, before: tuple[dict[str, float], dict[str, int]]
    ) -> tuple[dict[str, float], dict[str, int]]:
        prior_totals, prior_calls = before
        return (
            {stage: self.totals[stage] - prior_totals[stage] for stage in self.totals},
            {stage: self.calls[stage] - prior_calls[stage] for stage in self.calls},
        )

    def _timed(self, stage: str, original: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(original)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            _sync_mps()
            started = time.monotonic()
            output = original(*args, **kwargs)
            _sync_mps()
            self.totals[stage] += time.monotonic() - started
            self.calls[stage] += 1
            self.memory_samples.append(
                {"request": self.request, "stage": stage, **_memory_snapshot()}
            )
            return output

        return wrapper

    def _transformer_start(self, *_: object) -> None:
        _sync_mps()
        self._transformer_started = time.monotonic()

    def _transformer_end(self, *_: object) -> None:
        _sync_mps()
        self.totals["transformer"] += time.monotonic() - self._transformer_started
        self.calls["transformer"] += 1
        self.memory_samples.append(
            {"request": self.request, "stage": "transformer", **_memory_snapshot()}
        )


def _measure_request(
    pipeline: Any,
    timer: StageTimer,
    *,
    request: str,
    prompt: str,
    load_type: LoadType,
    height: int,
    width: int,
    frames: int,
    steps: int,
    guidance_scale: float,
    seed: int,
) -> dict[str, Any]:
    timer.request = request
    checkpoint = timer.checkpoint()
    start_memory = _memory_snapshot()
    _sync_mps()
    started = time.monotonic()
    output, fps = generate_video(
        prompt=prompt,
        pipeline=pipeline,
        output_type="pt",
        load_type=load_type,
        height=height,
        width=width,
        num_frames=frames,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        seed=seed,
    )
    _sync_mps()
    total_seconds = time.monotonic() - started
    stage_seconds, stage_calls = timer.delta(checkpoint)
    end_memory = _memory_snapshot()
    tensor_output = cast(torch.Tensor, output)
    shape = list(tensor_output.shape)
    del output
    gc.collect()
    return {
        "request": request,
        "total_seconds": total_seconds,
        "stage_seconds": stage_seconds,
        "stage_calls": stage_calls,
        "unattributed_seconds": max(0.0, total_seconds - sum(stage_seconds.values())),
        "output_shape": shape,
        "fps": fps,
        "memory_start": start_memory,
        "memory_end": end_memory,
    }


def _provenance() -> dict[str, Any]:
    git_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_revision": git_revision,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_git": torch.version.git_version,
        "diffusers": diffusers.__version__,
        "transformers": transformers.__version__,
        "accelerate": accelerate.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "fallback_enabled": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0",
    }


def _run_worker(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available")

    dtype = getattr(torch, args.dtype)
    load_started = time.monotonic()
    pipeline: Any = load_pipeline(args.model, dtype=dtype)
    load_seconds = time.monotonic() - load_started
    after_load = _memory_snapshot()

    prepare_started = time.monotonic()
    before_generation(pipeline, cast(LoadType, args.load_type))
    _sync_mps()
    prepare_seconds = time.monotonic() - prepare_started
    after_prepare = _memory_snapshot()

    with StageTimer(pipeline) as timer:
        cold = _measure_request(
            pipeline,
            timer,
            request="cold",
            prompt=args.prompt,
            load_type=cast(LoadType, args.load_type),
            height=args.height,
            width=args.width,
            frames=args.frames,
            steps=args.steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
        )
        warm = [
            _measure_request(
                pipeline,
                timer,
                request=f"warm_{index + 1}",
                prompt=args.prompt,
                load_type=cast(LoadType, args.load_type),
                height=args.height,
                width=args.width,
                frames=args.frames,
                steps=args.steps,
                guidance_scale=args.guidance_scale,
                seed=args.seed,
            )
            for index in range(args.warm_repeats)
        ]

    stage_summary = {
        stage: _summarize([run["stage_seconds"][stage] for run in warm])
        for stage in _ALL_STAGE_NAMES
    }
    memory_samples = [after_load, after_prepare, *timer.memory_samples]
    return {
        "status": "ok",
        "provenance": _provenance(),
        "config": {
            "model": args.model,
            "load_type": args.load_type,
            "dtype": args.dtype,
            "height": args.height,
            "width": args.width,
            "frames": args.frames,
            "steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "seed": args.seed,
            "warm_repeats": args.warm_repeats,
        },
        "load_seconds": load_seconds,
        "prepare_seconds": prepare_seconds,
        "memory_after_load": after_load,
        "memory_after_prepare": after_prepare,
        "cold_request": cold,
        "warm_requests": warm,
        "warm_summary": {
            "total": _summarize([run["total_seconds"] for run in warm]),
            "stages": stage_summary,
        },
        "stage_memory_samples": timer.memory_samples,
        "observed_max_mps_current_allocated_bytes": max(
            sample["current_allocated_bytes"] for sample in memory_samples
        ),
        "observed_max_mps_driver_allocated_bytes": max(
            sample["driver_allocated_bytes"] for sample in memory_samples
        ),
        "process_peak_rss_bytes": _peak_rss_bytes(),
    }


def _run_parent(args: argparse.Namespace) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    load_types = [value.strip() for value in args.load_types.split(",") if value.strip()]
    valid = {"mps", "cpu_model_offload", "sequential_cpu_offload"}
    invalid = set(load_types) - valid
    if invalid:
        raise ValueError(f"Unsupported load types: {sorted(invalid)}")

    def checkpoint_results() -> None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n")

    with tempfile.TemporaryDirectory(prefix="cogvideo-mps-bench-") as temporary_dir:
        for index, load_type in enumerate(load_types):
            worker_output = Path(temporary_dir) / f"{load_type}.json"
            command = [
                sys.executable,
                os.fspath(Path(__file__).resolve()),
                "--worker",
                "--worker-output",
                os.fspath(worker_output),
                "--load-type",
                load_type,
                "--model",
                args.model,
                "--dtype",
                args.dtype,
                "--prompt",
                args.prompt,
                "--height",
                str(args.height),
                "--width",
                str(args.width),
                "--frames",
                str(args.frames),
                "--steps",
                str(args.steps),
                "--guidance-scale",
                str(args.guidance_scale),
                "--seed",
                str(args.seed),
                "--warm-repeats",
                str(args.warm_repeats),
            ]
            environment = os.environ.copy()
            environment["COGKIT_DEVICE"] = "mps"
            environment["PYTORCH_ENABLE_MPS_FALLBACK"] = "1" if args.allow_fallback else "0"
            print(f"Running {load_type} in an isolated process", flush=True)
            try:
                completed = subprocess.run(
                    command,
                    check=False,
                    env=environment,
                    text=True,
                    timeout=args.timeout or None,
                )
            except subprocess.TimeoutExpired:
                results.append(
                    {
                        "status": "timeout",
                        "config": {"load_type": load_type},
                        "timeout_seconds": args.timeout,
                    }
                )
                checkpoint_results()
                continue
            if worker_output.exists():
                results.append(json.loads(worker_output.read_text()))
            else:
                results.append(
                    {
                        "status": "process_error",
                        "config": {"load_type": load_type},
                        "return_code": completed.returncode,
                    }
                )
            checkpoint_results()
            if index + 1 < len(load_types) and args.cooldown_seconds:
                print(f"Cooling down for {args.cooldown_seconds}s", flush=True)
                time.sleep(args.cooldown_seconds)
    return results


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="THUDM/CogVideoX-2b")
    parser.add_argument("--prompt", default="a paper boat drifting down a quiet stream")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--frames", type=int, default=9)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warm-repeats", type=int, default=5)
    parser.add_argument(
        "--load-types",
        default="mps,cpu_model_offload,sequential_cpu_offload",
        help="comma-separated modes; each runs in a fresh subprocess",
    )
    parser.add_argument("--allow-fallback", action="store_true")
    parser.add_argument("--timeout", type=int, default=0, help="per-mode timeout in seconds")
    parser.add_argument("--cooldown-seconds", type=int, default=30)
    parser.add_argument("--output", type=Path, default=Path("cogvideo_mps_benchmark.json"))
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--load-type", help=argparse.SUPPRESS)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.worker:
        if args.worker_output is None or args.load_type is None:
            raise ValueError("worker mode requires --worker-output and --load-type")
        try:
            result = _run_worker(args)
        except BaseException as error:
            result = {
                "status": "error",
                "config": {"load_type": args.load_type},
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
                "process_peak_rss_bytes": _peak_rss_bytes(),
            }
        args.worker_output.write_text(json.dumps(result, indent=2) + "\n")
        return 0 if result["status"] == "ok" else 1

    results = _run_parent(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Wrote {args.output}")
    return 0 if all(result["status"] == "ok" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
