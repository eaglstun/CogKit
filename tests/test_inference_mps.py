"""Inference device routing and opt-in CogView4 MPS smoke coverage.

Set COGKIT_RUN_MPS_INFERENCE=1 to run the real-model smoke test. Run this file under both
the pinned CogKit environment and an environment using the local PyTorch fork; the test
prints the exact torch version and import path for the result record.
"""

import os
import time
from typing import Any, cast

import numpy as np
import pytest
import torch

from cogkit.api.services import image_generation as image_generation_module
from cogkit.api.services.image_generation import ImageGenerationService
from cogkit.api.settings import APISettings
from cogkit.python import before_generation, generate_image
from cogkit.types import LoadType
from cogkit.utils import get_device, load_pipeline

INFERENCE_NOISE_MEAN_RTOL = 0.02
INFERENCE_LATENT_MEAN_RTOL = 0.05


class _FakeVAE:
    def enable_slicing(self) -> None:
        pass

    def enable_tiling(self) -> None:
        pass


class _FakePipeline:
    def __init__(self) -> None:
        self.vae = _FakeVAE()
        self.calls: list[tuple[str, object | None]] = []

    def remove_all_hooks(self) -> None:
        self.calls.append(("remove_all_hooks", None))

    def to(self, device: str) -> None:
        self.calls.append(("to", device))

    def enable_model_cpu_offload(self, device: torch.device) -> None:
        self.calls.append(("enable_model_cpu_offload", device))

    def enable_sequential_cpu_offload(self, device: torch.device) -> None:
        self.calls.append(("enable_sequential_cpu_offload", device))


def _pipeline() -> Any:
    return cast(Any, _FakePipeline())


def test_inference_device_auto_selects_mps(monkeypatch):
    monkeypatch.delenv("COGKIT_DEVICE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert get_device() == torch.device("mps")


def test_direct_mps_placement(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    pipeline = _pipeline()
    before_generation(pipeline, "mps")
    assert pipeline.calls[:2] == [("remove_all_hooks", None), ("to", "mps")]


def test_direct_mps_fails_loudly_when_unavailable(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="MPS is not available"):
        before_generation(_pipeline(), "mps")


@pytest.mark.parametrize(
    ("load_type", "method"),
    [
        ("cpu_model_offload", "enable_model_cpu_offload"),
        ("sequential_cpu_offload", "enable_sequential_cpu_offload"),
    ],
)
def test_cpu_offload_targets_mps(monkeypatch, load_type, method):
    monkeypatch.setenv("COGKIT_DEVICE", "mps")
    pipeline = _pipeline()
    before_generation(pipeline, load_type)
    assert (method, torch.device("mps")) in pipeline.calls


def test_cpu_offload_requires_accelerator(monkeypatch):
    monkeypatch.setenv("COGKIT_DEVICE", "cpu")
    with pytest.raises(RuntimeError, match="requires a CUDA or MPS accelerator"):
        before_generation(_pipeline(), "cpu_model_offload")


def test_api_threads_mps_load_type_to_generation(monkeypatch):
    pipeline = object()
    captured: dict[str, object] = {}

    monkeypatch.setattr(image_generation_module, "load_pipeline", lambda **kwargs: pipeline)

    def fake_generate_image(**kwargs):
        captured.update(kwargs)
        return np.zeros((1, 2, 2, 3), dtype=np.uint8)

    monkeypatch.setattr(image_generation_module, "generate_image", fake_generate_image)
    service = ImageGenerationService(APISettings(cogview4_path="model", offload_type="mps"))
    service.generate("cogview-4", "prompt", "2x2", 1)
    assert captured["load_type"] == "mps"


@pytest.mark.skipif(
    os.environ.get("COGKIT_RUN_MPS_INFERENCE") != "1",
    reason="set COGKIT_RUN_MPS_INFERENCE=1 to run the cached CogView4 smoke test",
)
def test_cogview4_real_model_mps_smoke(tmp_path):
    import diffusers
    import transformers

    if not torch.backends.mps.is_available():
        pytest.skip("MPS is not available")

    model = os.environ.get("COGKIT_MPS_MODEL_PATH", "THUDM/CogView4-6B")
    load_type = cast(LoadType, os.environ.get("COGKIT_MPS_LOAD_TYPE", "cpu_model_offload"))
    print(
        f"torch={torch.__version__} source={torch.__file__} model={model} "
        f"load_type={load_type} diffusers={diffusers.__version__} "
        f"transformers={transformers.__version__}"
    )
    pipeline = load_pipeline(model, dtype=torch.bfloat16)
    images = generate_image(
        prompt="a small red cube on a white background",
        pipeline=pipeline,
        load_type=load_type,
        height=512,
        width=512,
        num_inference_steps=1,
        seed=42,
    )
    assert len(images) == 1
    assert images[0].size == (512, 512)
    images[0].save(tmp_path / "mps-smoke.png")


@pytest.mark.skipif(
    os.environ.get("COGKIT_RUN_MPS_INFERENCE_PARITY") != "1",
    reason="set COGKIT_RUN_MPS_INFERENCE_PARITY=1 to run CPU-to-MPS latent parity",
)
def test_cogview4_one_step_cpu_mps_latent_parity():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is not available")

    model = os.environ.get("COGKIT_MPS_MODEL_PATH", "THUDM/CogView4-6B")
    pipeline = load_pipeline(model, dtype=torch.bfloat16)
    generator = torch.Generator(device="cpu").manual_seed(42)
    prompt_embeds = torch.randn((1, 16, 4096), generator=generator, dtype=torch.bfloat16)
    latents = torch.randn((1, 16, 64, 64), generator=generator, dtype=torch.float32)
    noise_predictions: list[torch.Tensor] = []
    scheduler_step = pipeline.scheduler.step

    def capture_noise_prediction(model_output, *args, **kwargs):
        noise_predictions.append(model_output.detach().float().cpu())
        return scheduler_step(model_output, *args, **kwargs)

    pipeline.scheduler.step = capture_noise_prediction

    started = time.monotonic()
    cpu_output = pipeline(
        prompt=None,
        prompt_embeds=prompt_embeds,
        latents=latents,
        guidance_scale=1.0,
        height=512,
        width=512,
        num_inference_steps=1,
        output_type="latent",
    ).images.float()
    cpu_seconds = time.monotonic() - started

    before_generation(pipeline, "cpu_model_offload")
    started = time.monotonic()
    mps_output = (
        pipeline(
            prompt=None,
            prompt_embeds=prompt_embeds.to("mps"),
            latents=latents,
            guidance_scale=1.0,
            height=512,
            width=512,
            num_inference_steps=1,
            output_type="latent",
        )
        .images.float()
        .cpu()
    )
    mps_seconds = time.monotonic() - started

    assert torch.isfinite(cpu_output).all()
    assert torch.isfinite(mps_output).all()
    cpu_noise, mps_noise = noise_predictions
    assert torch.isfinite(cpu_noise).all()
    assert torch.isfinite(mps_noise).all()
    noise_mean_relative_error = (mps_noise - cpu_noise).abs().mean() / cpu_noise.abs().mean()
    latent_mean_relative_error = (mps_output - cpu_output).abs().mean() / cpu_output.abs().mean()
    print(
        f"torch={torch.__version__} cpu={cpu_seconds:.2f}s mps={mps_seconds:.2f}s "
        f"noise_mean_relative_error={noise_mean_relative_error.item():.6f} "
        f"latent_mean_relative_error={latent_mean_relative_error.item():.6f}"
    )
    assert noise_mean_relative_error.item() < INFERENCE_NOISE_MEAN_RTOL
    assert latent_mean_relative_error.item() < INFERENCE_LATENT_MEAN_RTOL
