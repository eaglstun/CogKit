"""Runtime capability probe for bitsandbytes 4-bit (NF4) quantization.

`low_vram` (QLoRA) used to be gated on `device.type == "cuda"` because bitsandbytes
shipped no macOS build at all. A Metal/MPS build now exists, so the useful question is
no longer "is this CUDA" but "does bitsandbytes actually quantize on *this* device".

A platform string cannot answer that. An importable bitsandbytes with no working
backend for the device fails deep inside `from_pretrained`, long after the point where
a clear error is still cheap. So this runs the smallest real 4-bit round trip on the
target device and reports what happened.

The probe answers *availability*, not accuracy. Numerical correctness is the job of the
CPU-to-MPS parity gates, not of a guard that runs on every trainer startup.
"""

from functools import cache

import torch

__all__ = ["bnb_4bit_support", "require_bnb_4bit"]

_PROBE_NUMEL = 64  # one NF4 block at the smallest supported blocksize

_INSTALL_HINT = (
    "Install a bitsandbytes build with a backend for this device. On Apple Silicon that "
    "means a source build with Metal kernels: `cmake -DCOMPUTE_BACKEND=mps -S . && make` "
    "in a bitsandbytes checkout, then `pip install -e .`."
)


@cache
def bnb_4bit_support(device_type: str) -> tuple[bool, str]:
    """Report whether bitsandbytes can do NF4 quantization on `device_type`.

    Returns `(True, "")` when a 4-bit round trip completes on the device, otherwise
    `(False, reason)` with an actionable reason. Cached: the answer cannot change
    within a process, and the probe should not run once per module.
    """
    try:
        import bitsandbytes  # noqa: F401
    except ImportError as exc:
        return False, f"bitsandbytes is not installed ({exc}). {_INSTALL_HINT}"

    if device_type == "cuda" and not torch.cuda.is_available():
        return False, "device_type is 'cuda' but torch.cuda.is_available() is False."
    if device_type == "mps" and not torch.backends.mps.is_available():
        return False, "device_type is 'mps' but torch.backends.mps.is_available() is False."

    try:
        probe = torch.ones(_PROBE_NUMEL, dtype=torch.float32, device=device_type)
        packed, absmax = torch.ops.bitsandbytes.quantize_4bit(
            probe, _PROBE_NUMEL, "nf4", torch.uint8
        )
        restored = torch.ops.bitsandbytes.dequantize_4bit(
            packed, absmax, _PROBE_NUMEL, "nf4", list(probe.shape), torch.float32
        )
    except Exception as exc:  # noqa: BLE001 -- a probe must survive any backend failure mode
        return False, (
            f"bitsandbytes is installed but its 4-bit kernels do not run on '{device_type}': "
            f"{type(exc).__name__}: {exc}. {_INSTALL_HINT}"
        )

    # Backends differ on the returned shape (cpu hands back (1, N) for a (N,) input), so
    # check element count rather than shape -- this probe is about availability, not layout.
    if restored.numel() != probe.numel() or not bool(torch.isfinite(restored).all()):
        return False, (
            f"bitsandbytes 4-bit round trip on '{device_type}' returned a malformed result "
            f"(numel {restored.numel()} for a {probe.numel()}-element input, "
            f"finite={bool(torch.isfinite(restored).all())}). {_INSTALL_HINT}"
        )

    return True, ""


def require_bnb_4bit(device_type: str, feature: str = "low_vram (QLoRA via bitsandbytes NF4)") -> None:
    """Raise a `ValueError` naming `feature` unless NF4 works on `device_type`."""
    supported, reason = bnb_4bit_support(device_type)
    if not supported:
        raise ValueError(
            f"{feature} is not available on '{device_type}': {reason} "
            f"Set `low_vram: false` and use `mixed_precision: bf16` instead."
        )
