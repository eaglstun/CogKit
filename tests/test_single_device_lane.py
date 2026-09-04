"""Tests for the single-device (MPS/CPU) training lane.

See APPLE_METAL_PORT_PLAN.md. CPU-vs-MPS numerical parity for `compute_loss`
is covered separately (it needs model weights); these tests cover device
selection, config validation, and the loud-failure guards.
"""

import types

import pytest
import torch
from pydantic import ValidationError

import cogkit.finetune.base.base_trainer as base_trainer_module
from cogkit.finetune.base import BaseArgs
from cogkit.finetune.base.base_trainer import BaseTrainer
from cogkit.finetune.utils.dist import get_device
from cogkit.finetune.utils.memory import free_memory, get_memory_statistics


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


# ==============================================================================
# device selection
# ==============================================================================


def test_cogkit_device_override(monkeypatch):
    monkeypatch.setenv("COGKIT_DEVICE", "cpu")
    assert get_device().type == "cpu"


def test_get_device_auto(monkeypatch):
    monkeypatch.delenv("COGKIT_DEVICE", raising=False)
    monkeypatch.setenv("LOCAL_RANK", "0")
    device = get_device()
    if torch.cuda.is_available():
        assert device.type == "cuda"
    elif torch.backends.mps.is_available():
        assert device.type == "mps"
    else:
        assert device.type == "cpu"


# ==============================================================================
# args validation
# ==============================================================================


def test_single_strategy_accepted_for_lora():
    args = BaseArgs(**_base_args())
    assert args.strategy == "SINGLE"


def test_fsdp_strategy_rejected_for_lora():
    with pytest.raises(ValidationError, match="DDP.*SINGLE"):
        BaseArgs(**_base_args(strategy="FULL_SHARD"))


def test_low_vram_accepted_with_single():
    # QLoRA needs no collective, so the single-device lane is a valid host for it.
    args = BaseArgs(**_base_args(low_vram=True))
    assert args.low_vram and args.strategy == "SINGLE"


def test_low_vram_rejected_for_sft():
    with pytest.raises(ValidationError, match="low_vram"):
        BaseArgs(**_base_args(low_vram=True, training_type="sft", strategy="SINGLE"))


def test_offload_params_grads_rejected_with_single():
    with pytest.raises(ValidationError, match="offload_params_grads"):
        BaseArgs(**_base_args(strategy="SINGLE", offload_params_grads=True))


# ==============================================================================
# device-compat guards (must fail loudly, not silently)
# ==============================================================================


def _fake_trainer(strategy: str, low_vram: bool, device: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        uargs=types.SimpleNamespace(strategy=strategy, low_vram=low_vram),
        state=types.SimpleNamespace(device=torch.device(device)),
    )


def test_qlora_guard_errors_when_backend_unavailable(monkeypatch):
    # The guard is now a capability check, not a platform check: it must fail when
    # bitsandbytes cannot do 4-bit on this device, whatever the device is called.
    monkeypatch.setattr(
        base_trainer_module,
        "require_bnb_4bit",
        lambda device_type, *a, **k: (_ for _ in ()).throw(
            ValueError(f"bitsandbytes is not installed for {device_type}")
        ),
    )
    fake = _fake_trainer(strategy="SINGLE", low_vram=True, device="mps")
    with pytest.raises(ValueError, match="bitsandbytes"):
        BaseTrainer._check_device_compat(fake)


def test_qlora_guard_passes_when_backend_available(monkeypatch):
    monkeypatch.setattr(base_trainer_module, "require_bnb_4bit", lambda device_type, *a, **k: None)
    fake = _fake_trainer(strategy="SINGLE", low_vram=True, device="mps")
    BaseTrainer._check_device_compat(fake)


def test_qlora_guard_does_not_probe_when_low_vram_off(monkeypatch):
    # A bf16 run must never pay for (or trip over) the bitsandbytes probe.
    def _explode(*a, **k):
        raise AssertionError("require_bnb_4bit called with low_vram disabled")

    monkeypatch.setattr(base_trainer_module, "require_bnb_4bit", _explode)
    fake = _fake_trainer(strategy="SINGLE", low_vram=False, device="mps")
    BaseTrainer._check_device_compat(fake)


def test_fsdp_guard_errors_on_non_cuda():
    fake = _fake_trainer(strategy="FULL_SHARD", low_vram=False, device="mps")
    with pytest.raises(ValueError, match="SINGLE"):
        BaseTrainer._check_device_compat(fake)


def test_single_strategy_passes_guard_on_non_cuda():
    fake = _fake_trainer(strategy="SINGLE", low_vram=False, device="cpu")
    BaseTrainer._check_device_compat(fake)


# ==============================================================================
# memory utils
# ==============================================================================


def test_memory_statistics_cpu():
    stats = get_memory_statistics(torch.device("cpu"))
    assert stats["memory_allocated"] is None


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_memory_statistics_mps():
    stats = get_memory_statistics(torch.device("mps"))
    assert stats["memory_allocated"] is not None
    assert stats["max_memory_allocated"] is None  # MPS exposes no peak stats


def test_free_memory_does_not_crash():
    free_memory()
