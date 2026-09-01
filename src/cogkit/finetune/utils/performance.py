import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

import torch


def synchronize_device(device: torch.device) -> None:
    """Wait for queued accelerator work at an explicit profiling boundary."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


@dataclass
class TrainingProfiler:
    """Opt-in phase timer for asynchronous accelerator training.

    Timing an MPS or CUDA call without synchronizing measures command submission rather than
    execution. This helper synchronizes only while a requested profiling window is active; normal
    training therefore pays no synchronization cost.
    """

    device: torch.device
    warmup_steps: int = 1
    profile_steps: int = 0
    _samples: dict[str, list[float]] = field(
        default_factory=lambda: defaultdict(list), init=False, repr=False
    )

    @property
    def enabled(self) -> bool:
        return self.profile_steps > 0

    def should_profile(self, global_step: int) -> bool:
        return self.enabled and self.warmup_steps <= global_step < (
            self.warmup_steps + self.profile_steps
        )

    def record(self, phase: str, seconds: float) -> None:
        self._samples[phase].append(seconds)

    @contextmanager
    def measure(self, phase: str, enabled: bool) -> Iterator[None]:
        if not enabled:
            yield
            return

        synchronize_device(self.device)
        started = time.perf_counter()
        try:
            yield
        finally:
            synchronize_device(self.device)
            self.record(phase, time.perf_counter() - started)

    def summary(self) -> dict[str, dict[str, float | int | list[float]]]:
        result = {}
        for phase, samples in sorted(self._samples.items()):
            ordered = sorted(samples)
            midpoint = len(ordered) // 2
            if len(ordered) % 2:
                median = ordered[midpoint]
            else:
                median = (ordered[midpoint - 1] + ordered[midpoint]) / 2
            result[phase] = {
                "count": len(samples),
                "total_seconds": sum(samples),
                "mean_seconds": sum(samples) / len(samples),
                "median_seconds": median,
                "samples_seconds": samples,
            }
        return result
