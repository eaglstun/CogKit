"""Inference device routing and opt-in CogView4 MPS smoke coverage.

Set COGKIT_RUN_MPS_INFERENCE=1 to run the real-model smoke test. Run this file under both
the pinned CogKit environment and an environment using the local PyTorch fork; the test
prints the exact torch version and import path for the result record.

Set COGKIT_RUN_MPS_INFERENCE_PARITY=1 for CPU-to-MPS latent parity. Optional diagnostics:
COGKIT_MPS_PARITY_SIZE changes the square resolution (default 512), and
COGKIT_MPS_PARITY_STAGES=1 captures every transformer block plus first-block internals.

Set COGKIT_RUN_MPS_INFERENCE_COMPONENT_PARITY=1 for isolated real-model text-encoder and
VAE-decoder parity. COGKIT_MPS_COMPONENT_TEXT_LENGTH and COGKIT_MPS_COMPONENT_SIZE control
the prompt token limit and decoded square resolution (both default to 64).

Set COGKIT_RUN_MPS_INFERENCE_REAL_PROMPT_PARITY=1 to compose real prompt encoding with
the transformer and scheduler. Its text-length and resolution knobs use the same defaults.
COGKIT_MPS_REAL_PROMPT_DECODE=1 also runs VAE decode and checks the final tensor image.

Set COGKIT_RUN_MPS_INFERENCE_BENCHMARK=1 for the MPS step-cost benchmark. Unlike the parity
tests above, which time a single whole-pipeline call and are dominated by fixed overhead,
this one warms up in process, sweeps several step counts, and regresses wall time against
step count to separate fixed pipeline overhead from marginal per-step compute. Knobs:
COGKIT_MPS_BENCH_SIZE (default 512), COGKIT_MPS_BENCH_STEPS (comma list, default "1,4,8"),
COGKIT_MPS_BENCH_WARMUP (default 1), COGKIT_MPS_BENCH_REPEATS (default 5), and
COGKIT_MPS_BENCH_LOAD_TYPE (default "mps"; "cpu_model_offload" reproduces the parity tests'
offload behaviour, which shuttles submodules per call and inflates both cost and variance).
"""

import os
import statistics
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

INFERENCE_NOISE_MEAN_RTOL = 0.10
INFERENCE_NOISE_NORMALIZED_RMSE = 0.10
INFERENCE_NOISE_COSINE_MIN = 0.99
INFERENCE_LATENT_MEAN_RTOL = 0.05
INFERENCE_LATENT_NORMALIZED_RMSE = 0.05
INFERENCE_LATENT_COSINE_MIN = 0.995
INFERENCE_COMPOSED_LATENT_MEAN_RTOL = 0.075
INFERENCE_COMPOSED_LATENT_NORMALIZED_RMSE = 0.075
INFERENCE_COMPOSED_LATENT_COSINE_MIN = 0.995
INFERENCE_TEXT_COSINE_MIN = 0.999
INFERENCE_TEXT_SCALED_NORMALIZED_RMSE = 0.02
INFERENCE_TEXT_CONTEXT_MEAN_RTOL = 0.02
INFERENCE_TEXT_CONTEXT_NORMALIZED_RMSE = 0.02
INFERENCE_TEXT_CONTEXT_COSINE_MIN = 0.999
INFERENCE_VAE_MEAN_RTOL = 0.05
INFERENCE_VAE_NORMALIZED_RMSE = 0.05
INFERENCE_VAE_COSINE_MIN = 0.995
INFERENCE_IMAGE_MEAN_RTOL = 0.05
INFERENCE_IMAGE_NORMALIZED_RMSE = 0.05
INFERENCE_IMAGE_COSINE_MIN = 0.995


class _FakeVAE:
    def __init__(self) -> None:
        # Recorded rather than ignored: `_configure_vae_memory_saving` skips any method the
        # VAE does not expose, so a fake that silently absorbs everything would let the
        # memory-saving tests pass without the code doing anything.
        self.calls: list[str] = []

    def enable_slicing(self) -> None:
        self.calls.append("enable_slicing")

    def disable_slicing(self) -> None:
        self.calls.append("disable_slicing")

    def enable_tiling(self) -> None:
        self.calls.append("enable_tiling")

    def disable_tiling(self) -> None:
        self.calls.append("disable_tiling")


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


def _mean_relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return ((actual - expected).abs().mean() / expected.abs().mean()).item()


def _normalized_rmse(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return (((actual - expected).square().mean() / expected.square().mean()).sqrt()).item()


def _cosine_similarity(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual_flat = actual.double().flatten()
    expected_flat = expected.double().flatten()
    return (
        torch.dot(actual_flat, expected_flat) / (actual_flat.norm() * expected_flat.norm())
    ).item()


def _scaled_normalized_rmse(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual_flat = actual.double().flatten()
    expected_flat = expected.double().flatten()
    scale = torch.dot(actual_flat, expected_flat) / torch.dot(actual_flat, actual_flat)
    residual = scale * actual_flat - expected_flat
    return (residual.square().mean() / expected_flat.square().mean()).sqrt().item()


def _capture_transformer_stages(transformer) -> tuple[dict[str, list[torch.Tensor]], list[Any]]:
    stage_outputs: dict[str, list[torch.Tensor]] = {}
    handles = []

    def register(name: str, module) -> None:
        stage_outputs[name] = []

        def capture(_module, _inputs, output) -> None:
            hidden_states = output[0] if isinstance(output, tuple) else output
            stage_outputs[name].append(hidden_states.detach().float().cpu())

        handles.append(module.register_forward_hook(capture))

    register("patch_embed", transformer.patch_embed)
    register("time_condition_embed", transformer.time_condition_embed)
    for index, block in enumerate(transformer.transformer_blocks):
        register(f"block_{index:02d}", block)
    first_block = transformer.transformer_blocks[0]
    register("block_00_norm1", first_block.norm1)
    register("block_00_attention", first_block.attn1)
    register("block_00_norm2", first_block.norm2)
    register("block_00_ff_in_linear", first_block.ff.net[0].proj)
    register("block_00_ff_gelu", first_block.ff.net[0])
    register("block_00_ff_out_linear", first_block.ff.net[2])
    register("block_00_feed_forward", first_block.ff)
    register("norm_out", transformer.norm_out)
    register("proj_out", transformer.proj_out)
    return stage_outputs, handles


def _parity_metrics(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float, float]:
    return (
        _mean_relative_error(actual, expected),
        _normalized_rmse(actual, expected),
        _cosine_similarity(actual, expected),
    )


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


def test_vae_memory_saving_defaults_on(monkeypatch):
    # The default must stay conservative: the caller's memory budget is unknown.
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    pipeline = _pipeline()
    before_generation(pipeline, "mps")
    assert pipeline.vae.calls == ["enable_slicing", "enable_tiling"]


def test_vae_memory_saving_can_be_disabled(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    pipeline = _pipeline()
    before_generation(pipeline, "mps", vae_slicing=False, vae_tiling=False)
    assert pipeline.vae.calls == ["disable_slicing", "disable_tiling"]


def test_placement_is_not_repeated_for_the_same_load_type(monkeypatch):
    # Installing offload hooks twice is the cost this guard exists to avoid.
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    pipeline = _pipeline()
    before_generation(pipeline, "mps")
    before_generation(pipeline, "mps")
    assert pipeline.calls == [("remove_all_hooks", None), ("to", "mps")]


def test_vae_memory_saving_still_applies_on_a_repeat_call(monkeypatch):
    # Placement short-circuits on the second call; the VAE flags must not short-circuit
    # with it, or a caller could never change them between requests.
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    pipeline = _pipeline()
    before_generation(pipeline, "mps")
    pipeline.vae.calls.clear()
    before_generation(pipeline, "mps", vae_slicing=False, vae_tiling=False)
    assert pipeline.vae.calls == ["disable_slicing", "disable_tiling"]


def test_unset_vae_flags_do_not_reset_an_explicit_setting(monkeypatch):
    """Regression: `None` must mean "leave alone", not "enable".

    `generate_image`/`generate_video` forward only `load_type`, so every request re-enters
    `before_generation` with these unset. When `None` meant "enable", a caller who had
    turned tiling off had it silently turned back on by the next request -- which is how a
    benchmark's "tiling off" arm ended up measuring tiling on.
    """
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    pipeline = _pipeline()
    before_generation(pipeline, "mps", vae_slicing=False, vae_tiling=False)
    pipeline.vae.calls.clear()
    before_generation(pipeline, "mps")  # a subsequent request, flags not restated
    assert pipeline.vae.calls == []


def test_cpu_offload_requires_accelerator(monkeypatch):
    monkeypatch.setenv("COGKIT_DEVICE", "cpu")
    with pytest.raises(RuntimeError, match="requires a CUDA or MPS accelerator"):
        before_generation(_pipeline(), "cpu_model_offload")


def test_api_threads_mps_load_type_to_generation(monkeypatch):
    # The service now places the pipeline in __init__, so the stand-in has to be a
    # plausible pipeline rather than a bare object.
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    pipeline = _pipeline()
    captured: dict[str, object] = {}

    monkeypatch.setattr(image_generation_module, "load_pipeline", lambda **kwargs: pipeline)

    def fake_generate_image(**kwargs):
        captured.update(kwargs)
        return np.zeros((1, 2, 2, 3), dtype=np.uint8)

    monkeypatch.setattr(image_generation_module, "generate_image", fake_generate_image)
    service = ImageGenerationService(APISettings(cogview4_path="model", offload_type="mps"))
    service.generate("cogview-4", "prompt", "2x2", 1)
    assert captured["load_type"] == "mps"


def test_api_places_the_pipeline_at_startup(monkeypatch):
    """The first request should not be the one that pays for placement."""
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    pipeline = _pipeline()
    monkeypatch.setattr(image_generation_module, "load_pipeline", lambda **kwargs: pipeline)
    monkeypatch.setattr(
        image_generation_module,
        "generate_image",
        lambda **kwargs: np.zeros((1, 2, 2, 3), dtype=np.uint8),
    )

    ImageGenerationService(APISettings(cogview4_path="model", offload_type="mps"))
    assert pipeline.calls == [("remove_all_hooks", None), ("to", "mps")]
    assert pipeline.vae.calls == ["enable_slicing", "enable_tiling"]


def test_api_can_turn_vae_memory_saving_off(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    pipeline = _pipeline()
    monkeypatch.setattr(image_generation_module, "load_pipeline", lambda **kwargs: pipeline)
    monkeypatch.setattr(
        image_generation_module,
        "generate_image",
        lambda **kwargs: np.zeros((1, 2, 2, 3), dtype=np.uint8),
    )

    ImageGenerationService(
        APISettings(cogview4_path="model", offload_type="mps", vae_memory_saving=False)
    )
    assert pipeline.vae.calls == ["disable_slicing", "disable_tiling"]


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
        f"torch={torch.__version__} git={torch.version.git_version} "
        f"source={torch.__file__} model={model} "
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
    os.environ.get("COGKIT_RUN_MPS_INFERENCE_COMPONENT_PARITY") != "1",
    reason="set COGKIT_RUN_MPS_INFERENCE_COMPONENT_PARITY=1 for text/VAE parity",
)
def test_cogview4_text_encoder_and_vae_cpu_mps_parity():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is not available")

    model = os.environ.get("COGKIT_MPS_MODEL_PATH", "THUDM/CogView4-6B")
    text_length = int(os.environ.get("COGKIT_MPS_COMPONENT_TEXT_LENGTH", "64"))
    size = int(os.environ.get("COGKIT_MPS_COMPONENT_SIZE", "64"))
    if text_length <= 0:
        pytest.fail("COGKIT_MPS_COMPONENT_TEXT_LENGTH must be positive")
    if size % 32 != 0:
        pytest.fail("COGKIT_MPS_COMPONENT_SIZE must be divisible by 32")

    pipeline = load_pipeline(model, dtype=torch.bfloat16)
    prompt = "a small red cube on a white background"
    generator = torch.Generator(device="cpu").manual_seed(42)
    vae_latents = torch.randn(
        (1, pipeline.transformer.config.in_channels, size // 8, size // 8),
        generator=generator,
        dtype=torch.float32,
    )
    pipeline.vae.enable_slicing()
    pipeline.vae.enable_tiling()

    with torch.inference_mode():
        started = time.monotonic()
        cpu_prompt_embeds, _ = pipeline.encode_prompt(
            prompt,
            do_classifier_free_guidance=False,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
            max_sequence_length=text_length,
        )
        cpu_text_seconds = time.monotonic() - started
        started = time.monotonic()
        cpu_decoded = pipeline.vae.decode(
            vae_latents.to(dtype=pipeline.vae.dtype), return_dict=False
        )[0].float()
        cpu_vae_seconds = time.monotonic() - started

        before_generation(pipeline, "cpu_model_offload")
        started = time.monotonic()
        mps_prompt_embeds, _ = pipeline.encode_prompt(
            prompt,
            do_classifier_free_guidance=False,
            device=torch.device("mps"),
            dtype=torch.bfloat16,
            max_sequence_length=text_length,
        )
        mps_prompt_embeds = mps_prompt_embeds.float().cpu()
        mps_text_seconds = time.monotonic() - started
        started = time.monotonic()
        mps_decoded = (
            pipeline.vae.decode(
                vae_latents.to(device="mps", dtype=pipeline.vae.dtype), return_dict=False
            )[0]
            .float()
            .cpu()
        )
        mps_vae_seconds = time.monotonic() - started

    cpu_prompt_embeds = cpu_prompt_embeds.float()
    cpu_decoded = cpu_decoded.cpu()
    assert torch.isfinite(cpu_prompt_embeds).all()
    assert torch.isfinite(mps_prompt_embeds).all()
    assert torch.isfinite(cpu_decoded).all()
    assert torch.isfinite(mps_decoded).all()
    text_mre, text_nrmse, text_cosine = _parity_metrics(mps_prompt_embeds, cpu_prompt_embeds)
    text_scaled_nrmse = _scaled_normalized_rmse(mps_prompt_embeds, cpu_prompt_embeds)
    text_projection = pipeline.transformer.patch_embed.text_proj
    text_normalization = pipeline.transformer.transformer_blocks[0].norm1.norm_context
    cpu_text_context = text_normalization(
        text_projection(cpu_prompt_embeds.to(torch.bfloat16))
    ).float()
    mps_text_context = text_normalization(
        text_projection(mps_prompt_embeds.to(torch.bfloat16))
    ).float()
    context_mre, context_nrmse, context_cosine = _parity_metrics(mps_text_context, cpu_text_context)
    vae_mre, vae_nrmse, vae_cosine = _parity_metrics(mps_decoded, cpu_decoded)
    print(
        f"torch={torch.__version__} git={torch.version.git_version} "
        f"text_length={text_length} size={size} "
        f"cpu_text={cpu_text_seconds:.2f}s mps_text={mps_text_seconds:.2f}s "
        f"cpu_vae={cpu_vae_seconds:.2f}s mps_vae={mps_vae_seconds:.2f}s "
        f"text_mre={text_mre:.6f} text_nrmse={text_nrmse:.6f} "
        f"text_cosine={text_cosine:.6f} text_scaled_nrmse={text_scaled_nrmse:.6f} "
        f"context_mre={context_mre:.6f} context_nrmse={context_nrmse:.6f} "
        f"context_cosine={context_cosine:.6f} vae_mre={vae_mre:.6f} "
        f"vae_nrmse={vae_nrmse:.6f} vae_cosine={vae_cosine:.6f}"
    )
    assert text_cosine >= INFERENCE_TEXT_COSINE_MIN
    assert text_scaled_nrmse < INFERENCE_TEXT_SCALED_NORMALIZED_RMSE
    assert context_mre < INFERENCE_TEXT_CONTEXT_MEAN_RTOL
    assert context_nrmse < INFERENCE_TEXT_CONTEXT_NORMALIZED_RMSE
    assert context_cosine >= INFERENCE_TEXT_CONTEXT_COSINE_MIN
    assert vae_mre < INFERENCE_VAE_MEAN_RTOL
    assert vae_nrmse < INFERENCE_VAE_NORMALIZED_RMSE
    assert vae_cosine >= INFERENCE_VAE_COSINE_MIN


@pytest.mark.skipif(
    os.environ.get("COGKIT_RUN_MPS_INFERENCE_REAL_PROMPT_PARITY") != "1",
    reason="set COGKIT_RUN_MPS_INFERENCE_REAL_PROMPT_PARITY=1 for composed parity",
)
def test_cogview4_real_prompt_cpu_mps_latent_parity():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is not available")

    model = os.environ.get("COGKIT_MPS_MODEL_PATH", "THUDM/CogView4-6B")
    text_length = int(os.environ.get("COGKIT_MPS_COMPONENT_TEXT_LENGTH", "64"))
    size = int(os.environ.get("COGKIT_MPS_COMPONENT_SIZE", "64"))
    if text_length <= 0:
        pytest.fail("COGKIT_MPS_COMPONENT_TEXT_LENGTH must be positive")
    if size % 32 != 0:
        pytest.fail("COGKIT_MPS_COMPONENT_SIZE must be divisible by 32")
    decode = os.environ.get("COGKIT_MPS_REAL_PROMPT_DECODE") == "1"

    pipeline = load_pipeline(model, dtype=torch.bfloat16)
    generator = torch.Generator(device="cpu").manual_seed(42)
    latents = torch.randn(
        (1, pipeline.transformer.config.in_channels, size // 8, size // 8),
        generator=generator,
        dtype=torch.float32,
    )
    noise_predictions: list[torch.Tensor] = []
    scheduler_outputs: list[torch.Tensor] = []
    scheduler_step = pipeline.scheduler.step

    def capture_scheduler(model_output, *args, **kwargs):
        noise_predictions.append(model_output.detach().float().cpu())
        output = scheduler_step(model_output, *args, **kwargs)
        latent = output[0] if isinstance(output, tuple) else output.prev_sample
        scheduler_outputs.append(latent.detach().float().cpu())
        return output

    pipeline.scheduler.step = capture_scheduler
    call_kwargs = {
        "prompt": "a small red cube on a white background",
        "latents": latents,
        "guidance_scale": 1.0,
        "height": size,
        "width": size,
        "num_inference_steps": 1,
        "output_type": "pt" if decode else "latent",
        "max_sequence_length": text_length,
    }

    with torch.inference_mode():
        started = time.monotonic()
        cpu_output = pipeline(**call_kwargs).images.float()
        cpu_seconds = time.monotonic() - started
        before_generation(pipeline, "cpu_model_offload")
        started = time.monotonic()
        mps_output = pipeline(**call_kwargs).images.float().cpu()
        mps_seconds = time.monotonic() - started

    cpu_noise, mps_noise = noise_predictions
    cpu_scheduler, mps_scheduler = scheduler_outputs
    if not decode:
        assert torch.equal(cpu_output, cpu_scheduler)
        assert torch.equal(mps_output, mps_scheduler)
    assert torch.isfinite(cpu_noise).all()
    assert torch.isfinite(mps_noise).all()
    assert torch.isfinite(cpu_output).all()
    assert torch.isfinite(mps_output).all()
    noise_mre, noise_nrmse, noise_cosine = _parity_metrics(mps_noise, cpu_noise)
    latent_mre, latent_nrmse, latent_cosine = _parity_metrics(mps_scheduler, cpu_scheduler)
    print(
        f"torch={torch.__version__} git={torch.version.git_version} "
        f"text_length={text_length} size={size} "
        f"cpu={cpu_seconds:.2f}s mps={mps_seconds:.2f}s "
        f"noise_mre={noise_mre:.6f} noise_nrmse={noise_nrmse:.6f} "
        f"noise_cosine={noise_cosine:.6f} latent_mre={latent_mre:.6f} "
        f"latent_nrmse={latent_nrmse:.6f} latent_cosine={latent_cosine:.6f}"
    )
    if decode:
        assert torch.isfinite(cpu_output).all()
        assert torch.isfinite(mps_output).all()
        image_mre, image_nrmse, image_cosine = _parity_metrics(mps_output, cpu_output)
        print(
            f"image_mre={image_mre:.6f} image_nrmse={image_nrmse:.6f} "
            f"image_cosine={image_cosine:.6f}"
        )
        assert image_mre < INFERENCE_IMAGE_MEAN_RTOL
        assert image_nrmse < INFERENCE_IMAGE_NORMALIZED_RMSE
        assert image_cosine >= INFERENCE_IMAGE_COSINE_MIN
    assert noise_mre < INFERENCE_NOISE_MEAN_RTOL
    assert noise_nrmse < INFERENCE_NOISE_NORMALIZED_RMSE
    assert noise_cosine >= INFERENCE_NOISE_COSINE_MIN
    assert latent_mre < INFERENCE_COMPOSED_LATENT_MEAN_RTOL
    assert latent_nrmse < INFERENCE_COMPOSED_LATENT_NORMALIZED_RMSE
    assert latent_cosine >= INFERENCE_COMPOSED_LATENT_COSINE_MIN


@pytest.mark.skipif(
    os.environ.get("COGKIT_RUN_MPS_INFERENCE_PARITY") != "1",
    reason="set COGKIT_RUN_MPS_INFERENCE_PARITY=1 to run CPU-to-MPS latent parity",
)
def test_cogview4_one_step_cpu_mps_latent_parity():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is not available")

    model = os.environ.get("COGKIT_MPS_MODEL_PATH", "THUDM/CogView4-6B")
    size = int(os.environ.get("COGKIT_MPS_PARITY_SIZE", "512"))
    if size % 32 != 0:
        pytest.fail("COGKIT_MPS_PARITY_SIZE must be divisible by 32")
    pipeline = load_pipeline(model, dtype=torch.bfloat16)
    generator = torch.Generator(device="cpu").manual_seed(42)
    prompt_embeds = torch.randn((1, 16, 4096), generator=generator, dtype=torch.bfloat16)
    latents = torch.randn((1, 16, size // 8, size // 8), generator=generator, dtype=torch.float32)
    noise_predictions: list[torch.Tensor] = []
    stage_outputs: dict[str, list[torch.Tensor]] = {}
    stage_handles: list[Any] = []
    if os.environ.get("COGKIT_MPS_PARITY_STAGES") == "1":
        stage_outputs, stage_handles = _capture_transformer_stages(pipeline.transformer)
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
        height=size,
        width=size,
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
            height=size,
            width=size,
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
    noise_mean_relative_error = _mean_relative_error(mps_noise, cpu_noise)
    latent_mean_relative_error = _mean_relative_error(mps_output, cpu_output)
    noise_normalized_rmse = _normalized_rmse(mps_noise, cpu_noise)
    latent_normalized_rmse = _normalized_rmse(mps_output, cpu_output)
    noise_cosine_similarity = _cosine_similarity(mps_noise, cpu_noise)
    latent_cosine_similarity = _cosine_similarity(mps_output, cpu_output)
    for handle in stage_handles:
        handle.remove()
    print(
        f"torch={torch.__version__} git={torch.version.git_version} size={size} "
        f"cpu={cpu_seconds:.2f}s mps={mps_seconds:.2f}s "
        f"noise_mean_relative_error={noise_mean_relative_error:.6f} "
        f"latent_mean_relative_error={latent_mean_relative_error:.6f} "
        f"noise_normalized_rmse={noise_normalized_rmse:.6f} "
        f"latent_normalized_rmse={latent_normalized_rmse:.6f} "
        f"noise_cosine_similarity={noise_cosine_similarity:.6f} "
        f"latent_cosine_similarity={latent_cosine_similarity:.6f}"
    )
    if stage_outputs:
        stage_errors = []
        for name, outputs in stage_outputs.items():
            assert len(outputs) >= 2 and len(outputs) % 2 == 0, (
                f"expected paired CPU and MPS captures for {name}"
            )
            mps_start = len(outputs) // 2
            stage_errors.append((name, _mean_relative_error(outputs[mps_start], outputs[0])))
        ff_in_outputs = stage_outputs["block_00_ff_in_linear"]
        ff_gelu_outputs = stage_outputs["block_00_ff_gelu"]
        mps_ff_in = ff_in_outputs[len(ff_in_outputs) // 2].to(torch.bfloat16)
        cpu_gelu_from_mps_input = torch.nn.functional.gelu(mps_ff_in, approximate="tanh").float()
        gelu_input_sensitivity = _mean_relative_error(cpu_gelu_from_mps_input, ff_gelu_outputs[0])
        mps_gelu = ff_gelu_outputs[len(ff_gelu_outputs) // 2]
        gelu_kernel_error = _mean_relative_error(mps_gelu, cpu_gelu_from_mps_input)
        print("transformer_stage_mean_relative_errors=")
        print(" ".join(f"{name}:{error:.6f}" for name, error in stage_errors))
        print(
            f"block_00_gelu_input_sensitivity={gelu_input_sensitivity:.6f} "
            f"block_00_gelu_kernel_error={gelu_kernel_error:.6f}"
        )
    assert noise_mean_relative_error < INFERENCE_NOISE_MEAN_RTOL
    assert noise_normalized_rmse < INFERENCE_NOISE_NORMALIZED_RMSE
    assert noise_cosine_similarity >= INFERENCE_NOISE_COSINE_MIN
    assert latent_mean_relative_error < INFERENCE_LATENT_MEAN_RTOL
    assert latent_normalized_rmse < INFERENCE_LATENT_NORMALIZED_RMSE
    assert latent_cosine_similarity >= INFERENCE_LATENT_COSINE_MIN


@pytest.mark.skipif(
    os.environ.get("COGKIT_RUN_MPS_INFERENCE_BENCHMARK") != "1",
    reason="set COGKIT_RUN_MPS_INFERENCE_BENCHMARK=1 to run the MPS step-cost benchmark",
)
def test_cogview4_mps_step_cost_benchmark():
    """Measure marginal MPS cost per denoising step, separated from fixed overhead.

    A single-step whole-pipeline timing cannot distinguish a change in kernel performance
    from a change in setup cost, and its run-to-run spread exceeds the effect sizes worth
    measuring. Sweeping step counts and taking the slope isolates the compute that actually
    scales with work done.
    """
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is not available")

    model = os.environ.get("COGKIT_MPS_MODEL_PATH", "THUDM/CogView4-6B")
    size = int(os.environ.get("COGKIT_MPS_BENCH_SIZE", "512"))
    if size % 32 != 0:
        pytest.fail("COGKIT_MPS_BENCH_SIZE must be divisible by 32")
    step_counts = sorted({int(part) for part in os.environ["COGKIT_MPS_BENCH_STEPS"].split(",")}) \
        if os.environ.get("COGKIT_MPS_BENCH_STEPS") else [1, 4, 8]
    if any(steps < 1 for steps in step_counts):
        pytest.fail("COGKIT_MPS_BENCH_STEPS entries must be >= 1")
    warmup = int(os.environ.get("COGKIT_MPS_BENCH_WARMUP", "1"))
    repeats = int(os.environ.get("COGKIT_MPS_BENCH_REPEATS", "5"))
    if repeats < 1:
        pytest.fail("COGKIT_MPS_BENCH_REPEATS must be >= 1")
    load_type = cast(LoadType, os.environ.get("COGKIT_MPS_BENCH_LOAD_TYPE", "mps"))

    pipeline = load_pipeline(model, dtype=torch.bfloat16)
    before_generation(pipeline, load_type)

    generator = torch.Generator(device="cpu").manual_seed(42)
    prompt_embeds = torch.randn((1, 16, 4096), generator=generator, dtype=torch.bfloat16).to("mps")
    latents = torch.randn((1, 16, size // 8, size // 8), generator=generator, dtype=torch.float32)

    def run(steps: int) -> torch.Tensor:
        output = pipeline(
            prompt=None,
            prompt_embeds=prompt_embeds,
            latents=latents.clone(),
            guidance_scale=1.0,
            height=size,
            width=size,
            num_inference_steps=steps,
            output_type="latent",
        ).images
        # MPS dispatch is asynchronous; without this the timer measures queue submission.
        torch.mps.synchronize()
        return output

    # Discarded: absorbs Metal shader compilation and first-touch allocation.
    for _ in range(warmup):
        run(min(step_counts))

    samples: dict[int, list[float]] = {}
    for steps in step_counts:
        timings: list[float] = []
        for _ in range(repeats):
            torch.mps.synchronize()
            started = time.monotonic()
            output = run(steps)
            timings.append(time.monotonic() - started)
        assert torch.isfinite(output.float()).all()
        samples[steps] = sorted(timings)

    medians = {steps: statistics.median(values) for steps, values in samples.items()}

    # Least-squares slope of wall time against step count: intercept is the fixed
    # pipeline overhead, slope is the per-step compute that optimization work moves.
    fixed_overhead = float("nan")
    marginal_per_step = float("nan")
    if len(step_counts) >= 2:
        mean_steps = statistics.mean(step_counts)
        mean_time = statistics.mean(medians[steps] for steps in step_counts)
        denominator = sum((steps - mean_steps) ** 2 for steps in step_counts)
        marginal_per_step = (
            sum((steps - mean_steps) * (medians[steps] - mean_time) for steps in step_counts)
            / denominator
        )
        fixed_overhead = mean_time - marginal_per_step * mean_steps

    per_step_counts = " ".join(
        f"steps{steps}_median={medians[steps]:.3f}s "
        f"steps{steps}_min={samples[steps][0]:.3f}s "
        f"steps{steps}_max={samples[steps][-1]:.3f}s"
        for steps in step_counts
    )
    print(
        f"bench torch={torch.__version__} git={torch.version.git_version} "
        f"load_type={load_type} size={size} warmup={warmup} repeats={repeats} "
        f"{per_step_counts} "
        f"fixed_overhead={fixed_overhead:.3f}s marginal_per_step={marginal_per_step:.3f}s"
    )
