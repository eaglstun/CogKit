"""CPU-vs-MPS numerical parity for `compute_loss` (APPLE_METAL_PORT_PLAN.md §4).

CPU is the oracle. The same cached batch and *injected* noise/timestep run on
both devices in separate processes (a same-seed `torch.Generator` yields
different sequences on cpu vs mps, so a shared seed is NOT enough), then the
loss and the pre-loss `noise_pred` tensor are compared within bf16 tolerance.

Heavy: needs the CogView4-6B weights (cached by the Phase-1 smoke run) and the
precomputed `.cache/` under quickstart/data/t2i. Skipped when either is absent.

Run directly for one device:
    python tests/test_mps_cpu_parity.py --device cpu --out /tmp/cpu.pt --fixtures /tmp/fx.pt
"""

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG = REPO_ROOT / "quickstart" / "scripts" / "t2i" / "config_mps.yaml"
QLORA_CONFIG = REPO_ROOT / "quickstart" / "scripts" / "t2i" / "config_mps_qlora.yaml"
DATA_CACHE = REPO_ROOT / "quickstart" / "data" / "t2i" / "train" / ".cache"

# bf16 has ~7 mantissa bits; different accumulation orders across backends make
# elementwise drift expected. These are documented, deliberately loose bounds.
LOSS_RTOL = 2e-2
NOISE_PRED_MEAN_REL = 5e-2

# NF4-vs-bf16 bounds. Quantizing the base transformer to 4 bits moves `noise_pred`
# ~21% elementwise by design, so an elementwise bound would only encode that number.
# These bound what QLoRA actually needs: the same training signal (measured 1.3% loss
# difference) and a prediction pointing the same way (measured cosine 0.979).
QLORA_LOSS_RTOL = 5e-2
QLORA_COSINE_MIN = 0.95

FIXED_TIMESTEP = 500
NOISE_SEED = 1234


def _weights_cached() -> bool:
    hub = Path.home() / ".cache" / "huggingface" / "hub" / "models--THUDM--CogView4-6B"
    return hub.exists()


def _run_device(
    device: str,
    out_path: Path,
    fixtures_path: Path,
    port: int,
    overfit_steps: int = 0,
    backward: bool = False,
    config: "Path" = CONFIG,
) -> None:
    env = os.environ.copy()
    env.update(
        {
            "COGKIT_DEVICE": device,
            "PYTORCH_ENABLE_MPS_FALLBACK": "1",
            "RANK": "0",
            "LOCAL_RANK": "0",
            "WORLD_SIZE": "1",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    subprocess.run(
        [
            sys.executable,
            __file__,
            "--device",
            device,
            "--out",
            str(out_path),
            "--fixtures",
            str(fixtures_path),
            "--overfit-steps",
            str(overfit_steps),
            "--config",
            str(config),
            *(["--backward"] if backward else []),
        ],
        env=env,
        cwd=REPO_ROOT / "quickstart" / "scripts" / "t2i",
        check=True,
        timeout=7200,  # the CPU-oracle backward (6B, checkpointed, CPU bf16) is legitimately slow
    )


def test_compute_loss_cpu_mps_parity(tmp_path):
    """Forward parity (loss + noise_pred), CPU vs MPS. ~6 min.

    Set COGKIT_PARITY_BACKWARD=1 to also compare the LoRA grad norm — WARNING:
    the CPU-oracle backward of the 6B transformer takes >2h on an M4 Max (slow
    CPU bf16 autograd kernels); backward correctness is normally covered by the
    much cheaper test_mps_can_learn below.
    """
    import pytest
    import torch

    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    if not _weights_cached():
        pytest.skip("CogView4-6B weights not in HF cache (run the Phase-1 smoke run first)")
    if not DATA_CACHE.exists():
        pytest.skip("precompute cache missing (run the Phase-1 smoke run first)")

    backward = os.environ.get("COGKIT_PARITY_BACKWARD") == "1"

    fixtures = tmp_path / "fixtures.pt"
    torch.save({"noise_seed": NOISE_SEED, "timestep": FIXED_TIMESTEP}, fixtures)

    cpu_out, mps_out = tmp_path / "cpu.pt", tmp_path / "mps.pt"
    _run_device("cpu", cpu_out, fixtures, port=29511, backward=backward)
    _run_device("mps", mps_out, fixtures, port=29512, backward=backward)

    cpu = torch.load(cpu_out, weights_only=True)
    mps = torch.load(mps_out, weights_only=True)

    loss_cpu, loss_mps = cpu["loss"].item(), mps["loss"].item()
    rel = abs(loss_mps - loss_cpu) / max(abs(loss_cpu), 1e-8)
    print(f"loss cpu={loss_cpu:.6f} mps={loss_mps:.6f} rel_diff={rel:.4f}")
    assert rel < LOSS_RTOL, f"loss diverges: cpu={loss_cpu} mps={loss_mps} rel={rel}"

    np_cpu = cpu["noise_pred"].float()
    np_mps = mps["noise_pred"].float()
    mean_rel = (np_mps - np_cpu).abs().mean() / np_cpu.abs().mean().clamp_min(1e-8)
    print(f"noise_pred mean_rel_diff={mean_rel.item():.4f}")
    assert mean_rel < NOISE_PRED_MEAN_REL, f"noise_pred diverges: mean_rel={mean_rel.item()}"

    if backward:
        gn_cpu, gn_mps = cpu["grad_norm"].item(), mps["grad_norm"].item()
        gn_rel = abs(gn_mps - gn_cpu) / max(abs(gn_cpu), 1e-8)
        print(f"grad_norm cpu={gn_cpu:.6f} mps={gn_mps:.6f} rel_diff={gn_rel:.4f}")
        assert gn_rel < NOISE_PRED_MEAN_REL, f"backward diverges: grad_norm rel={gn_rel}"


def test_mps_can_learn(tmp_path):
    """Overfit-one-batch on MPS: with batch/noise/timestep held fixed, repeated
    optimizer steps must drive the loss down — the direct test that the MPS
    backward pass + optimizer actually learn (the running loss of a real run is
    too timestep-noisy to show a trend over a short smoke run)."""
    import pytest
    import torch

    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    if not _weights_cached():
        pytest.skip("CogView4-6B weights not in HF cache (run the Phase-1 smoke run first)")
    if not DATA_CACHE.exists():
        pytest.skip("precompute cache missing (run the Phase-1 smoke run first)")

    fixtures = tmp_path / "fixtures.pt"
    torch.save({"noise_seed": NOISE_SEED, "timestep": FIXED_TIMESTEP}, fixtures)
    out = tmp_path / "overfit.pt"
    _run_device("mps", out, fixtures, port=29513, overfit_steps=15)

    result = torch.load(out, weights_only=True)
    losses = result["losses"].tolist()
    print("overfit losses:", [round(x, 4) for x in losses])
    assert all(x == x for x in losses), f"NaN in overfit losses: {losses}"
    # LoRA @ lr 1e-4 drops ~9% over 15 steps on this fixture; what indicts a broken
    # backward is the absence of a steady downward trend, not a big drop
    assert losses[-1] == min(losses), f"loss not trending down under overfit: {losses}"
    assert losses[-1] < losses[0] * 0.95, f"loss did not decrease under overfit: {losses}"


def test_qlora_matches_bf16_on_fixed_batch(tmp_path):
    """NF4 base weights must not change what the model is computing.

    This compares bf16 against NF4 on MPS with the batch, noise and timestep held
    fixed, so the only variable is the weight quantization. It is NOT a device test:
    the divergence measured here is NF4 quantization error and would look the same on
    CUDA. `test_qlora_cpu_mps_parity` is the one that indicts the Metal kernels.

    Elementwise `noise_pred` moves ~21% under NF4 -- that is what quantizing a 6B
    transformer to 4 bits does, and the reason QLoRA trains adapters to compensate.
    What must hold is that the prediction still points the same way (cosine) and that
    the training signal the optimizer sees is essentially unchanged (loss).
    """
    import pytest
    import torch

    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    if not _weights_cached():
        pytest.skip("CogView4-6B weights not in HF cache (run the Phase-1 smoke run first)")
    if not DATA_CACHE.exists():
        pytest.skip("precompute cache missing (run the Phase-1 smoke run first)")

    fixtures = tmp_path / "fixtures.pt"
    torch.save({"noise_seed": NOISE_SEED, "timestep": FIXED_TIMESTEP}, fixtures)

    bf16_out, nf4_out = tmp_path / "bf16.pt", tmp_path / "nf4.pt"
    _run_device("mps", bf16_out, fixtures, port=29521, config=CONFIG)
    _run_device("mps", nf4_out, fixtures, port=29522, config=QLORA_CONFIG)

    bf16 = torch.load(bf16_out, weights_only=True)
    nf4 = torch.load(nf4_out, weights_only=True)

    loss_bf16, loss_nf4 = bf16["loss"].item(), nf4["loss"].item()
    rel = abs(loss_nf4 - loss_bf16) / max(abs(loss_bf16), 1e-8)
    print(f"loss bf16={loss_bf16:.6f} nf4={loss_nf4:.6f} rel_diff={rel:.4f}")
    assert rel < QLORA_LOSS_RTOL, f"NF4 loss diverges from bf16: {loss_bf16} vs {loss_nf4}"

    pred_bf16 = bf16["noise_pred"].float().flatten()
    pred_nf4 = nf4["noise_pred"].float().flatten()
    cos = torch.nn.functional.cosine_similarity(pred_bf16, pred_nf4, dim=0).item()
    print(f"noise_pred cosine={cos:.6f}")
    assert cos > QLORA_COSINE_MIN, f"NF4 prediction points elsewhere: cosine={cos}"


def test_qlora_can_learn(tmp_path):
    """Overfit-one-batch under QLoRA on MPS.

    The frozen base is NF4 and only the LoRA adapters carry gradients, so this is the
    direct test that the backward through `Linear4bit` on Metal produces a usable
    signal -- a dequantize path that silently returned garbage gradients would still
    let a training run finish, and would still print a plausible-looking loss.
    """
    import pytest
    import torch

    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    if not _weights_cached():
        pytest.skip("CogView4-6B weights not in HF cache (run the Phase-1 smoke run first)")
    if not DATA_CACHE.exists():
        pytest.skip("precompute cache missing (run the Phase-1 smoke run first)")

    fixtures = tmp_path / "fixtures.pt"
    torch.save({"noise_seed": NOISE_SEED, "timestep": FIXED_TIMESTEP}, fixtures)
    out = tmp_path / "overfit_qlora.pt"
    _run_device("mps", out, fixtures, port=29523, overfit_steps=15, config=QLORA_CONFIG)

    result = torch.load(out, weights_only=True)
    losses = result["losses"].tolist()
    print("qlora overfit losses:", [round(x, 4) for x in losses])
    assert torch.isfinite(result["losses"]).all(), f"non-finite QLoRA overfit losses: {losses}"
    assert losses[-1] == min(losses), f"loss not trending down under QLoRA overfit: {losses}"
    assert losses[-1] < losses[0] * 0.95, f"loss did not decrease under QLoRA overfit: {losses}"


def _runner(
    device: str,
    out_path: str,
    fixtures_path: str,
    overfit_steps: int = 0,
    backward: bool = False,
    config_path: str = str(CONFIG),
) -> None:
    """Build the trainer on one device, run compute_loss on the first cached
    batch with injected noise/timestep, save loss + noise_pred. With
    overfit_steps > 0, instead take that many AdamW steps on the fixed batch
    and save the per-step loss sequence."""
    import torch
    import torch.distributed as dist

    fixtures = torch.load(fixtures_path, weights_only=True)

    from cogkit.finetune import get_model_cls

    trainer_cls = get_model_cls("cogview4-6b", "lora", False)
    trainer = trainer_cls(config_path)

    trainer.prepare_models()
    trainer.prepare_dataset()
    trainer.prepare_trainable_parameters()
    trainer.prepare_model()

    # Numerically identical either way; a speed/memory trade per device. On MPS the
    # recompute is the bottleneck (disable). On CPU, storing every activation of a 6B
    # forward swaps the box — keep checkpointing on and accept the slower backward.
    if device != "cpu" and trainer.components.transformer.is_gradient_checkpointing:
        trainer.components.transformer.disable_gradient_checkpointing()

    batch = next(iter(trainer.train_data_loader))

    dev = trainer.state.device
    # inject deterministic randomness: identical noise (generated on CPU) + fixed timestep
    latent_shape = batch["encoded_image"].shape
    gen = torch.Generator("cpu").manual_seed(int(fixtures["noise_seed"]))
    fixed_noise = torch.randn(latent_shape, generator=gen, dtype=torch.float32)
    fixed_t = torch.tensor([int(fixtures["timestep"])], device=dev)

    trainer.get_timestep = lambda batch_size, num_train_timesteps: fixed_t.repeat(batch_size)

    orig_randn_like = torch.randn_like

    def _fixed_randn_like(t, **kwargs):
        return fixed_noise.to(device=t.device, dtype=t.dtype)

    torch.randn_like = _fixed_randn_like

    # capture the transformer output (the pre-loss tensor)
    captured = {}
    transformer = trainer.components.transformer
    orig_forward = transformer.forward

    def _capturing_forward(*args, **kwargs):
        out = orig_forward(*args, **kwargs)
        captured["noise_pred"] = out[0].detach().to("cpu", torch.float32)
        return out

    transformer.forward = _capturing_forward

    if overfit_steps > 0:
        optimizer = torch.optim.AdamW(
            [p for p in transformer.parameters() if p.requires_grad], lr=1e-4
        )
        losses = []
        for _ in range(overfit_steps):
            optimizer.zero_grad()
            loss = trainer.compute_loss(batch)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        torch.randn_like = orig_randn_like
        torch.save({"losses": torch.tensor(losses)}, out_path)
        print(f"[{device}] overfit {losses[0]:.4f} -> {losses[-1]:.4f}")
        dist.destroy_process_group()
        return

    result = {}
    if backward:
        loss = trainer.compute_loss(batch)
        loss.backward()
        trainable = [p for p in transformer.parameters() if p.requires_grad and p.grad is not None]
        grad_norm = torch.norm(torch.stack([p.grad.detach().float().norm() for p in trainable]))
        result["grad_norm"] = grad_norm.to("cpu")
        print(f"[{device}] grad_norm={grad_norm.item():.6f}")
    else:
        with torch.no_grad():
            loss = trainer.compute_loss(batch)

    torch.randn_like = orig_randn_like
    result["loss"] = loss.detach().to("cpu", torch.float32)
    result["noise_pred"] = captured["noise_pred"]
    torch.save(result, out_path)
    print(f"[{device}] loss={loss.item():.6f} -> {out_path}")
    dist.destroy_process_group()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--fixtures", required=True)
    parser.add_argument("--overfit-steps", type=int, default=0)
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--config", default=str(CONFIG))
    cli_args = parser.parse_args()
    _runner(
        cli_args.device,
        cli_args.out,
        cli_args.fixtures,
        cli_args.overfit_steps,
        cli_args.backward,
        cli_args.config,
    )
