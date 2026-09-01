import time

import pytest
import torch
from pydantic import ValidationError

from cogkit.finetune.base import BaseArgs
from cogkit.finetune.utils.performance import TrainingProfiler


def _base_args(**overrides) -> dict:
    args = {
        "name4train": "test",
        "model_path": "THUDM/CogView4-6B",
        "model_name": "cogview4-6b",
        "data_root": ".",
        "training_type": "lora",
        "strategy": "SINGLE",
        "train_epochs": 1,
        "checkpointing_steps": 5,
        "checkpointing_limit": 2,
        "batch_size": 1,
        "mixed_precision": "bf16",
        "validation_steps": None,
    }
    args.update(overrides)
    return args


def test_profile_window_excludes_warmup_and_stops_after_requested_steps():
    profiler = TrainingProfiler(torch.device("cpu"), warmup_steps=2, profile_steps=3)

    assert [profiler.should_profile(step) for step in range(7)] == [
        False,
        False,
        True,
        True,
        True,
        False,
        False,
    ]


def test_profiler_records_json_serializable_statistics():
    profiler = TrainingProfiler(torch.device("cpu"), warmup_steps=0, profile_steps=1)

    with profiler.measure("forward", enabled=True):
        time.sleep(0.001)
    profiler.record("forward", 0.003)

    summary = profiler.summary()["forward"]
    assert summary["count"] == 2
    assert summary["median_seconds"] == pytest.approx(sum(summary["samples_seconds"]) / 2)
    assert summary["total_seconds"] >= 0.004


def test_disabled_measurement_does_not_synchronize(monkeypatch):
    synchronize_calls = []
    monkeypatch.setattr(torch.mps, "synchronize", lambda: synchronize_calls.append(True))
    profiler = TrainingProfiler(torch.device("mps"), profile_steps=0)

    with profiler.measure("forward", enabled=False):
        pass

    assert synchronize_calls == []
    assert profiler.summary() == {}


def test_mps_measurement_synchronizes_at_both_boundaries(monkeypatch):
    synchronize_calls = []
    monkeypatch.setattr(torch.mps, "synchronize", lambda: synchronize_calls.append(True))
    profiler = TrainingProfiler(torch.device("mps"), warmup_steps=0, profile_steps=1)

    with profiler.measure("forward", enabled=True):
        pass

    assert synchronize_calls == [True, True]


@pytest.mark.parametrize("field", ["profile_warmup_steps", "profile_steps"])
def test_profile_step_counts_must_be_non_negative(field):
    with pytest.raises(ValidationError):
        BaseArgs(**_base_args(**{field: -1}))
