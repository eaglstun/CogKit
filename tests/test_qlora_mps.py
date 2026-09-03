"""QLoRA (bitsandbytes NF4) support on the single-device lane.

Two things are checked here, and they are different questions:

1. The capability probe in `cogkit.finetune.utils.quantization` reports honestly.
   `low_vram` used to be gated on `device.type == "cuda"`; it is now gated on whether
   bitsandbytes can actually run 4-bit kernels on the target device.
2. `Linear4bit` -- the layer QLoRA actually trains through -- produces the same forward
   output *and the same input gradients* on MPS as on CPU. A quantized forward that runs
   is not a quantized forward that is correct, and QLoRA needs the backward too.

Requires a bitsandbytes with a working backend; every test skips cleanly without one.
"""

import pytest
import torch

from cogkit.finetune.utils.quantization import bnb_4bit_support, require_bnb_4bit

bnb = pytest.importorskip("bitsandbytes", reason="bitsandbytes is not installed")

MPS_AVAILABLE = torch.backends.mps.is_available()
mps_only = pytest.mark.skipif(not MPS_AVAILABLE, reason="MPS not available")


# ==============================================================================
# capability probe
# ==============================================================================


def test_probe_reports_cuda_unavailable_without_cuda():
    if torch.cuda.is_available():
        pytest.skip("CUDA is available on this host")
    supported, reason = bnb_4bit_support("cuda")
    assert not supported
    assert "cuda" in reason


def test_probe_reason_is_empty_when_supported():
    supported, reason = bnb_4bit_support("cpu")
    if not supported:
        pytest.skip(f"bitsandbytes has no working cpu backend here: {reason}")
    assert reason == ""


def test_probe_is_cached():
    # The probe allocates and quantizes; it must not do that once per caller.
    assert bnb_4bit_support("cpu") is bnb_4bit_support("cpu")


def test_require_raises_with_actionable_message():
    if torch.cuda.is_available():
        pytest.skip("CUDA is available on this host")
    with pytest.raises(ValueError, match="low_vram"):
        require_bnb_4bit("cuda")


@mps_only
def test_probe_reports_mps_supported():
    supported, reason = bnb_4bit_support("mps")
    assert supported, f"bitsandbytes 4-bit does not run on MPS here: {reason}"


# ==============================================================================
# Linear4bit numerical parity, CPU oracle vs MPS
# ==============================================================================


def _build_linear4bit(weight: torch.Tensor, device: str, compute_dtype, double_quant: bool):
    from bitsandbytes.nn import Linear4bit, Params4bit

    out_features, in_features = weight.shape
    linear = Linear4bit(
        in_features,
        out_features,
        bias=False,
        compute_dtype=compute_dtype,
        quant_type="nf4",
        compress_statistics=double_quant,
    )
    linear.weight = Params4bit(
        weight.clone(), requires_grad=False, quant_type="nf4", compress_statistics=double_quant
    )
    return linear.to(device)


def _forward_backward(weight, x, device: str, compute_dtype, double_quant: bool):
    linear = _build_linear4bit(weight, device, compute_dtype, double_quant)
    inp = x.to(device).clone().requires_grad_(True)
    out = linear(inp)
    # A scalar that depends on every output element, so the backward is not sparse.
    loss = out.float().pow(2).mean()
    loss.backward()
    return out.detach().float().cpu(), inp.grad.detach().float().cpu(), loss.item()


@mps_only
@pytest.mark.parametrize("double_quant", [False, True], ids=["single_quant", "double_quant"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_linear4bit_forward_backward_parity(dtype, double_quant):
    supported, reason = bnb_4bit_support("mps")
    if not supported:
        pytest.skip(reason)

    torch.manual_seed(0)
    in_features = out_features = 512
    weight = torch.randn(out_features, in_features, dtype=torch.float32) * 0.02
    x = torch.randn(1, 64, in_features, dtype=dtype) * 0.5

    out_cpu, grad_cpu, loss_cpu = _forward_backward(weight, x, "cpu", dtype, double_quant)
    out_mps, grad_mps, loss_mps = _forward_backward(weight, x, "mps", dtype, double_quant)

    assert torch.isfinite(out_mps).all() and torch.isfinite(grad_mps).all()

    # Both devices dequantize the same NF4 codes, so this is not a "close enough for
    # 4-bit" tolerance -- it is a tolerance on the matmul, well under quantization error.
    out_scale = out_cpu.abs().mean().clamp(min=1e-12)
    grad_scale = grad_cpu.abs().mean().clamp(min=1e-12)
    out_rel = ((out_mps - out_cpu).abs().mean() / out_scale).item()
    grad_rel = ((grad_mps - grad_cpu).abs().mean() / grad_scale).item()

    assert out_rel < 1e-3, f"forward mean rel diff {out_rel:.2e}"
    assert grad_rel < 1e-3, f"input-grad mean rel diff {grad_rel:.2e}"
    assert abs(loss_mps - loss_cpu) / max(abs(loss_cpu), 1e-12) < 1e-3


@mps_only
def test_nf4_roundtrip_matches_cpu_codes_bit_exactly():
    supported, reason = bnb_4bit_support("mps")
    if not supported:
        pytest.skip(reason)

    torch.manual_seed(1234)
    a = torch.randn(4096, dtype=torch.float32)

    packed_cpu, absmax_cpu = torch.ops.bitsandbytes.quantize_4bit(a, 64, "nf4", torch.uint8)
    packed_mps, absmax_mps = torch.ops.bitsandbytes.quantize_4bit(
        a.to("mps"), 64, "nf4", torch.uint8
    )

    # NF4 packing is a pure function of the input; anything other than bit-exact here
    # means the two devices disagree about which weights the model actually holds.
    assert torch.equal(packed_mps.cpu(), packed_cpu)
    assert torch.equal(absmax_mps.cpu(), absmax_cpu)
