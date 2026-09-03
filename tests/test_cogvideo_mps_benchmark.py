import sys
from types import SimpleNamespace

import pytest
import torch

from tools.benchmark_cogvideo_mps import StageTimer, _peak_rss_bytes, _summarize


def test_benchmark_summary_reports_spread() -> None:
    summary = _summarize([1.0, 2.0, 3.0])

    assert summary == {
        "count": 3,
        "median_seconds": 2.0,
        "mean_seconds": 2.0,
        "min_seconds": 1.0,
        "max_seconds": 3.0,
        "standard_deviation_seconds": pytest.approx(0.816496580927726),
        "coefficient_of_variation": pytest.approx(0.408248290463863),
    }


def test_benchmark_summary_handles_no_samples() -> None:
    assert _summarize([]) == {"count": 0}


def test_peak_rss_is_normalized_to_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    class Usage:
        ru_maxrss = 123

    monkeypatch.setattr(
        "tools.benchmark_cogvideo_mps.resource.getrusage", lambda _: Usage()
    )
    monkeypatch.setattr(sys, "platform", "linux")

    assert _peak_rss_bytes() == 123 * 1024


def test_transformer_timer_survives_forward_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Transformer(torch.nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return value + 1

    def identity(value=None, **_):
        return value

    pipeline = SimpleNamespace(
        encode_prompt=identity,
        transformer=Transformer(),
        scheduler=SimpleNamespace(step=identity),
        vae=SimpleNamespace(decode=identity),
        video_processor=SimpleNamespace(postprocess_video=identity),
    )
    monkeypatch.setattr("tools.benchmark_cogvideo_mps._sync_mps", lambda: None)
    monkeypatch.setattr(
        "tools.benchmark_cogvideo_mps._memory_snapshot",
        lambda: {
            "current_allocated_bytes": 0,
            "driver_allocated_bytes": 0,
            "process_peak_rss_bytes": 0,
        },
    )

    with StageTimer(pipeline) as timer:
        pipeline.transformer(torch.tensor(1))
        pipeline.transformer.forward = lambda value: value + 2
        pipeline.transformer(torch.tensor(1))

    assert timer.calls["transformer"] == 2
    assert len(timer.memory_samples) == 2
