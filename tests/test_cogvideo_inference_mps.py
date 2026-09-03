"""Hardware-free coverage for CogVideo inference routing and MPS preparation."""

import importlib
import os
import time
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from click.testing import CliRunner
from diffusers import CogVideoXDPMScheduler
from PIL import Image

from cogkit.cli import inference as inference_command
from cogkit.python.generation import util as generation_util
from cogkit.python.generation import video as video_generation
from cogkit.types import GenerationMode
from cogkit.utils import load_pipeline

inference_module = importlib.import_module("cogkit.cli.inference")

COGVIDEO_NOISE_MEAN_RTOL = 0.01
COGVIDEO_NOISE_NORMALIZED_RMSE = 0.01
COGVIDEO_NOISE_COSINE_MIN = 0.9999
COGVIDEO_LATENT_MEAN_RTOL = 0.01
COGVIDEO_LATENT_NORMALIZED_RMSE = 0.01
COGVIDEO_LATENT_COSINE_MIN = 0.9999


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


def _parity_metrics(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float, float]:
    return (
        _mean_relative_error(actual, expected),
        _normalized_rmse(actual, expected),
        _cosine_similarity(actual, expected),
    )


def _capture_cogvideo_transformer_stages(
    transformer: Any,
) -> tuple[dict[str, list[torch.Tensor]], list[Any]]:
    outputs: dict[str, list[torch.Tensor]] = {}
    handles = []

    def register(name: str, module: torch.nn.Module) -> None:
        outputs[name] = []

        def capture(_module, _inputs, output) -> None:
            hidden_states = output[0] if isinstance(output, tuple) else output
            outputs[name].append(hidden_states.detach().float().cpu())

        handles.append(module.register_forward_hook(capture))

    blocks = transformer.transformer_blocks
    register("patch_embed", transformer.patch_embed)
    register("block_00", blocks[0])
    register(f"block_{len(blocks) // 2:02d}", blocks[len(blocks) // 2])
    register(f"block_{len(blocks) - 1:02d}", blocks[-1])
    register("norm_final", transformer.norm_final)
    register("proj_out", transformer.proj_out)
    return outputs, handles


class _FakeVAE:
    def __init__(self) -> None:
        self.slicing_calls = 0
        self.tiling_calls = 0

    def enable_slicing(self) -> None:
        self.slicing_calls += 1

    def enable_tiling(self) -> None:
        self.tiling_calls += 1


class _OriginalScheduler:
    config = {"name": "original"}


class _FakeDPMScheduler:
    from_config_calls = 0

    @classmethod
    def from_config(cls, config: dict[str, str], **kwargs: str) -> "_FakeDPMScheduler":
        assert config == {"name": "original"}
        assert kwargs == {"timestep_spacing": "trailing"}
        cls.from_config_calls += 1
        return cls()


class _FakeVideoPipeline:
    def __init__(self) -> None:
        self.scheduler: Any = _OriginalScheduler()
        self.vae = _FakeVAE()
        self.transformer = torch.nn.Module()
        self.transformer.register_buffer(
            "pos_embedding", torch.ones(2, dtype=torch.float64), persistent=False
        )
        self.components = {"transformer": self.transformer}
        self.calls: list[tuple[str, object | None]] = []

    def remove_all_hooks(self) -> None:
        self.calls.append(("remove_all_hooks", None))

    def to(self, device: str) -> None:
        self.calls.append(("to", device))

    def enable_model_cpu_offload(self, device: torch.device) -> None:
        self.calls.append(("enable_model_cpu_offload", device))

    def enable_sequential_cpu_offload(self, device: torch.device) -> None:
        self.calls.append(("enable_sequential_cpu_offload", device))


class _CallableVideoPipeline:
    def __init__(self) -> None:
        self.kwargs: dict[str, object] = {}

    def __call__(self, **kwargs: object) -> tuple[torch.Tensor]:
        self.kwargs = kwargs
        return (torch.zeros((1, 9, 3, 8, 8)),)


class _FrameConfigPipeline:
    def __init__(self, patch_size_t: int | None) -> None:
        config = SimpleNamespace(patch_size_t=patch_size_t, sample_frames=49)
        self.transformer = SimpleNamespace(config=config)


def test_cogvideo_preparation_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeDPMScheduler.from_config_calls = 0
    monkeypatch.setattr(generation_util, "TVideoPipeline", _FakeVideoPipeline)
    monkeypatch.setattr(generation_util, "CogVideoXDPMScheduler", _FakeDPMScheduler)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    pipeline = _FakeVideoPipeline()

    generation_util.before_generation(cast(Any, pipeline), "mps")
    generation_util.before_generation(cast(Any, pipeline), "mps")

    assert isinstance(pipeline.scheduler, _FakeDPMScheduler)
    assert _FakeDPMScheduler.from_config_calls == 1
    assert pipeline.calls == [("remove_all_hooks", None), ("to", "mps")]
    assert pipeline.vae.slicing_calls == 1
    assert pipeline.vae.tiling_calls == 1
    assert pipeline.transformer.pos_embedding.dtype == torch.float32


@pytest.mark.parametrize(
    ("patch_size_t", "valid_frames", "fps", "invalid_frames", "error"),
    [
        (None, 9, 8, 10, "8N\\+1"),
        (2, 17, 16, 18, "16N\\+1"),
    ],
)
def test_cogvideo_frame_contracts(
    monkeypatch: pytest.MonkeyPatch,
    patch_size_t: int | None,
    valid_frames: int,
    fps: int,
    invalid_frames: int,
    error: str,
) -> None:
    monkeypatch.setattr(generation_util, "TVideoPipeline", _FrameConfigPipeline)
    monkeypatch.setattr(generation_util, "CogVideoXPipeline", _FrameConfigPipeline)
    monkeypatch.setattr(generation_util, "CogVideoXImageToVideoPipeline", _FrameConfigPipeline)
    pipeline = _FrameConfigPipeline(patch_size_t)

    assert generation_util.guess_frames(cast(Any, pipeline), valid_frames) == (valid_frames, fps)
    with pytest.raises(AssertionError, match=error):
        generation_util.guess_frames(cast(Any, pipeline), invalid_frames)


@pytest.mark.parametrize(
    ("mode", "with_image"),
    [
        (GenerationMode.TextToVideo, False),
        (GenerationMode.ImageToVideo, True),
    ],
)
def test_generate_video_routes_short_seeded_workload(
    monkeypatch: pytest.MonkeyPatch,
    mode: GenerationMode,
    with_image: bool,
) -> None:
    pipeline = _CallableVideoPipeline()
    image = Image.new("RGB", (8, 8)) if with_image else None
    prepared: list[tuple[object, str]] = []
    monkeypatch.setattr(video_generation, "guess_generation_mode", lambda **_: mode)
    monkeypatch.setattr(video_generation, "guess_resolution", lambda *_: (480, 720))
    monkeypatch.setattr(video_generation, "guess_frames", lambda *_: (9, 8))
    monkeypatch.setattr(
        video_generation,
        "before_generation",
        lambda target, load_type: prepared.append((target, load_type)),
    )

    output, fps = video_generation.generate_video(
        prompt="a paper boat on a stream",
        pipeline=pipeline,  # type: ignore[arg-type]
        input_image=image,
        output_type="pt",
        load_type="mps",
        height=480,
        width=720,
        num_frames=9,
        num_inference_steps=1,
        guidance_scale=1.5,
        seed=123,
    )

    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, 9, 3, 8, 8)
    assert fps == 8
    assert prepared == [(pipeline, "mps")]
    assert pipeline.kwargs["num_frames"] == 9
    assert pipeline.kwargs["num_inference_steps"] == 1
    assert pipeline.kwargs["guidance_scale"] == 1.5
    generator = pipeline.kwargs["generator"]
    assert isinstance(generator, torch.Generator)
    assert generator.device.type == "cpu"
    assert generator.initial_seed() == 123
    if with_image:
        assert pipeline.kwargs["image"] is image
    else:
        assert "image" not in pipeline.kwargs


def test_cli_forwards_short_video_controls(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    pipeline = object()
    captured: dict[str, object] = {}
    monkeypatch.setattr(inference_module, "load_pipeline", lambda *args: pipeline)
    monkeypatch.setattr(
        inference_module,
        "guess_generation_mode",
        lambda **kwargs: GenerationMode.TextToVideo,
    )

    def fake_generate_video(**kwargs: object) -> tuple[list[object], int]:
        captured.update(kwargs)
        return [object()], 8

    monkeypatch.setattr(inference_module, "generate_video", fake_generate_video)
    monkeypatch.setattr(inference_module, "export_to_video", lambda *args, **kwargs: None)
    output_file = tmp_path / "short.mp4"

    result = CliRunner().invoke(
        inference_command,
        [
            "a paper boat on a stream",
            "THUDM/CogVideoX-2b",
            "--load_type",
            "mps",
            "--num_frames",
            "9",
            "--num_inference_steps",
            "1",
            "--guidance_scale",
            "1.5",
            "--output_file",
            str(output_file),
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["pipeline"] is pipeline
    assert captured["load_type"] == "mps"
    assert captured["num_frames"] == 9
    assert captured["num_inference_steps"] == 1
    assert captured["guidance_scale"] == 1.5


def test_cli_keeps_video_frames_out_of_image_generation(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    pipeline = object()
    captured: dict[str, object] = {}
    monkeypatch.setattr(inference_module, "load_pipeline", lambda *args: pipeline)
    monkeypatch.setattr(
        inference_module,
        "guess_generation_mode",
        lambda **kwargs: GenerationMode.TextToImage,
    )

    def fake_generate_image(**kwargs: object) -> list[Image.Image]:
        captured.update(kwargs)
        return [Image.new("RGB", (8, 8))]

    monkeypatch.setattr(inference_module, "generate_image", fake_generate_image)
    output_file = tmp_path / "image.png"

    result = CliRunner().invoke(
        inference_command,
        [
            "a paper boat on a stream",
            "THUDM/CogView4-6B",
            "--num_frames",
            "9",
            "--output_file",
            str(output_file),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "num_frames" not in captured
    assert captured["guidance_scale"] == 3.5


@pytest.mark.skipif(
    os.environ.get("COGKIT_RUN_COGVIDEO_MPS_COMPONENT_PARITY") != "1",
    reason="set COGKIT_RUN_COGVIDEO_MPS_COMPONENT_PARITY=1 for T5 parity",
)
def test_cogvideox_text_encoder_cpu_mps_parity() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is not available")

    model = os.environ.get("COGKIT_COGVIDEO_MODEL_PATH", "THUDM/CogVideoX-2b")
    pipeline: Any = load_pipeline(model, dtype=torch.bfloat16)
    prompt = "a paper boat drifting down a quiet stream"
    text_length = pipeline.transformer.config.max_text_seq_length

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
        print(f"CPU T5 complete in {cpu_text_seconds:.2f}s", flush=True)

        generation_util.before_generation(pipeline, "cpu_model_offload")
        torch.mps.synchronize()
        started = time.monotonic()
        mps_prompt_embeds, _ = pipeline.encode_prompt(
            prompt,
            do_classifier_free_guidance=False,
            device=torch.device("mps"),
            dtype=torch.bfloat16,
            max_sequence_length=text_length,
        )
        torch.mps.synchronize()
        mps_text_seconds = time.monotonic() - started
        mps_prompt_embeds = mps_prompt_embeds.float().cpu()

    cpu_prompt_embeds = cpu_prompt_embeds.float().cpu()
    for tensor in (cpu_prompt_embeds, mps_prompt_embeds):
        assert torch.isfinite(tensor).all()

    text_mre, text_nrmse, text_cosine = _parity_metrics(mps_prompt_embeds, cpu_prompt_embeds)
    text_scaled_nrmse = _scaled_normalized_rmse(mps_prompt_embeds, cpu_prompt_embeds)
    print(
        f"torch={torch.__version__} git={torch.version.git_version} model={model} "
        f"text_length={text_length} "
        f"cpu_text={cpu_text_seconds:.2f}s mps_text={mps_text_seconds:.2f}s "
        f"text_mre={text_mre:.6f} text_nrmse={text_nrmse:.6f} "
        f"text_cosine={text_cosine:.6f} text_scaled_nrmse={text_scaled_nrmse:.6f}"
    )

    assert text_scaled_nrmse < 0.055
    assert text_cosine >= 0.998


@pytest.mark.skipif(
    os.environ.get("COGKIT_RUN_COGVIDEO_MPS_VAE_PARITY") != "1",
    reason="set COGKIT_RUN_COGVIDEO_MPS_VAE_PARITY=1 for VAE parity",
)
def test_cogvideox_vae_cpu_mps_parity() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is not available")

    model = os.environ.get("COGKIT_COGVIDEO_MODEL_PATH", "THUDM/CogVideoX-2b")
    height = int(os.environ.get("COGKIT_COGVIDEO_VAE_HEIGHT", "32"))
    width = int(os.environ.get("COGKIT_COGVIDEO_VAE_WIDTH", "32"))
    frames = int(os.environ.get("COGKIT_COGVIDEO_VAE_FRAMES", "1"))
    if height % 16 != 0 or width % 16 != 0:
        pytest.fail("CogVideo VAE height and width must be divisible by 16")
    if (frames - 1) % 8 != 0:
        pytest.fail("CogVideoX 1.0 VAE frames must be 8N+1")

    pipeline: Any = load_pipeline(model, dtype=torch.bfloat16)
    pipeline.vae.enable_slicing()
    pipeline.vae.enable_tiling()
    generator = torch.Generator(device="cpu").manual_seed(42)
    latent_frames = (frames - 1) // pipeline.vae_scale_factor_temporal + 1
    vae_latents = torch.randn(
        (
            1,
            latent_frames,
            pipeline.transformer.config.out_channels,
            height // pipeline.vae_scale_factor_spatial,
            width // pipeline.vae_scale_factor_spatial,
        ),
        generator=generator,
        dtype=torch.bfloat16,
    )

    with torch.inference_mode():
        started = time.monotonic()
        cpu_decoded = pipeline.decode_latents(vae_latents).float().cpu()
        cpu_vae_seconds = time.monotonic() - started
        print(f"CPU VAE complete in {cpu_vae_seconds:.2f}s", flush=True)

        generation_util.before_generation(pipeline, "cpu_model_offload")
        torch.mps.synchronize()
        started = time.monotonic()
        mps_decoded = pipeline.decode_latents(vae_latents.to("mps")).float()
        torch.mps.synchronize()
        mps_vae_seconds = time.monotonic() - started
        mps_decoded = mps_decoded.cpu()

    for tensor in (cpu_decoded, mps_decoded):
        assert torch.isfinite(tensor).all()

    vae_mre, vae_nrmse, vae_cosine = _parity_metrics(mps_decoded, cpu_decoded)
    print(
        f"torch={torch.__version__} git={torch.version.git_version} model={model} "
        f"shape={height}x{width}x{frames} "
        f"cpu_vae={cpu_vae_seconds:.2f}s mps_vae={mps_vae_seconds:.2f}s "
        f"vae_mre={vae_mre:.6f} vae_nrmse={vae_nrmse:.6f} vae_cosine={vae_cosine:.6f}"
    )

    assert vae_mre < 0.01
    assert vae_nrmse < 0.01
    assert vae_cosine >= 0.9999


@pytest.mark.skipif(
    os.environ.get("COGKIT_RUN_COGVIDEO_MPS_PARITY") != "1",
    reason="set COGKIT_RUN_COGVIDEO_MPS_PARITY=1 for CogVideo transformer parity",
)
def test_cogvideox_transformer_scheduler_cpu_mps_parity() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is not available")

    model = os.environ.get("COGKIT_COGVIDEO_MODEL_PATH", "THUDM/CogVideoX-2b")
    height = int(os.environ.get("COGKIT_COGVIDEO_PARITY_HEIGHT", "480"))
    width = int(os.environ.get("COGKIT_COGVIDEO_PARITY_WIDTH", "720"))
    frames = int(os.environ.get("COGKIT_COGVIDEO_PARITY_FRAMES", "9"))
    if height % 16 != 0 or width % 16 != 0:
        pytest.fail("CogVideo parity height and width must be divisible by 16")
    if (frames - 1) % 8 != 0:
        pytest.fail("CogVideoX 1.0 parity frames must be 8N+1")

    pipeline: Any = load_pipeline(model, dtype=torch.bfloat16)
    pipeline.scheduler = CogVideoXDPMScheduler.from_config(
        pipeline.scheduler.config, timestep_spacing="trailing"
    )
    generator = torch.Generator(device="cpu").manual_seed(42)
    prompt_embeds = torch.randn(
        (1, pipeline.transformer.config.max_text_seq_length, 4096),
        generator=generator,
        dtype=torch.bfloat16,
    )
    latent_frames = (frames - 1) // pipeline.vae_scale_factor_temporal + 1
    latents = torch.randn(
        (
            1,
            latent_frames,
            pipeline.transformer.config.in_channels,
            height // pipeline.vae_scale_factor_spatial,
            width // pipeline.vae_scale_factor_spatial,
        ),
        generator=generator,
        dtype=torch.bfloat16,
    )
    noise_predictions: list[torch.Tensor] = []
    scheduler_outputs: list[torch.Tensor] = []
    scheduler_step = pipeline.scheduler.step

    def capture_scheduler(model_output, *args, **kwargs):
        noise_predictions.append(model_output.detach().float().cpu())
        output = scheduler_step(model_output, *args, **kwargs)
        scheduler_output = output[0] if isinstance(output, tuple) else output.prev_sample
        scheduler_outputs.append(scheduler_output.detach().float().cpu())
        return output

    pipeline.scheduler.step = capture_scheduler
    stage_outputs: dict[str, list[torch.Tensor]] = {}
    stage_handles: list[Any] = []
    if os.environ.get("COGKIT_COGVIDEO_PARITY_STAGES") == "1":
        stage_outputs, stage_handles = _capture_cogvideo_transformer_stages(pipeline.transformer)

    call_kwargs = {
        "prompt": None,
        "latents": latents,
        "guidance_scale": 1.0,
        "height": height,
        "width": width,
        "num_frames": frames,
        "num_inference_steps": 1,
        "output_type": "latent",
    }
    with torch.inference_mode():
        started = time.monotonic()
        cpu_output = pipeline(prompt_embeds=prompt_embeds, **call_kwargs).frames.float()
        cpu_seconds = time.monotonic() - started

        generation_util.before_generation(pipeline, "cpu_model_offload")
        torch.mps.synchronize()
        started = time.monotonic()
        mps_output = pipeline(prompt_embeds=prompt_embeds.to("mps"), **call_kwargs).frames.float()
        torch.mps.synchronize()
        mps_seconds = time.monotonic() - started
        mps_output = mps_output.cpu()

    for handle in stage_handles:
        handle.remove()

    assert len(noise_predictions) == 2
    assert len(scheduler_outputs) == 2
    cpu_noise, mps_noise = noise_predictions
    cpu_scheduler, mps_scheduler = scheduler_outputs
    for tensor in (cpu_noise, mps_noise, cpu_scheduler, mps_scheduler, cpu_output, mps_output):
        assert torch.isfinite(tensor).all()

    noise_mre, noise_nrmse, noise_cosine = _parity_metrics(mps_noise, cpu_noise)
    latent_mre, latent_nrmse, latent_cosine = _parity_metrics(mps_scheduler, cpu_scheduler)
    output_mre, output_nrmse, output_cosine = _parity_metrics(mps_output, cpu_output)
    print(
        f"torch={torch.__version__} git={torch.version.git_version} model={model} "
        f"shape={height}x{width}x{frames} cpu={cpu_seconds:.2f}s mps={mps_seconds:.2f}s "
        f"noise_mre={noise_mre:.6f} noise_nrmse={noise_nrmse:.6f} "
        f"noise_cosine={noise_cosine:.6f} latent_mre={latent_mre:.6f} "
        f"latent_nrmse={latent_nrmse:.6f} latent_cosine={latent_cosine:.6f} "
        f"output_mre={output_mre:.6f} output_nrmse={output_nrmse:.6f} "
        f"output_cosine={output_cosine:.6f}"
    )
    if stage_outputs:
        stage_metrics = []
        for name, outputs in stage_outputs.items():
            assert len(outputs) == 2, f"expected CPU and MPS captures for {name}"
            stage_metrics.append((name, _parity_metrics(outputs[1], outputs[0])))
        print(
            "transformer_stage_metrics="
            + " ".join(
                f"{name}:mre={metrics[0]:.6f},nrmse={metrics[1]:.6f},cos={metrics[2]:.6f}"
                for name, metrics in stage_metrics
            )
        )

    assert noise_mre < COGVIDEO_NOISE_MEAN_RTOL
    assert noise_nrmse < COGVIDEO_NOISE_NORMALIZED_RMSE
    assert noise_cosine >= COGVIDEO_NOISE_COSINE_MIN
    assert latent_mre < COGVIDEO_LATENT_MEAN_RTOL
    assert latent_nrmse < COGVIDEO_LATENT_NORMALIZED_RMSE
    assert latent_cosine >= COGVIDEO_LATENT_COSINE_MIN
    assert output_mre < COGVIDEO_LATENT_MEAN_RTOL
    assert output_nrmse < COGVIDEO_LATENT_NORMALIZED_RMSE
    assert output_cosine >= COGVIDEO_LATENT_COSINE_MIN


@pytest.mark.skipif(
    os.environ.get("COGKIT_RUN_COGVIDEO_MPS_INFERENCE") != "1",
    reason="set COGKIT_RUN_COGVIDEO_MPS_INFERENCE=1 to run the CogVideoX MPS smoke test",
)
def test_cogvideox_real_model_mps_smoke() -> None:
    import diffusers
    import transformers

    if not torch.backends.mps.is_available():
        pytest.skip("MPS is not available")

    model = os.environ.get("COGKIT_COGVIDEO_MODEL_PATH", "THUDM/CogVideoX-2b")
    load_type = os.environ.get("COGKIT_COGVIDEO_LOAD_TYPE", "cpu_model_offload")
    print(
        f"torch={torch.__version__} git={torch.version.git_version} "
        f"source={torch.__file__} model={model} load_type={load_type} "
        f"diffusers={diffusers.__version__} transformers={transformers.__version__}"
    )
    pipeline = load_pipeline(model, dtype=torch.bfloat16)

    output, fps = video_generation.generate_video(
        prompt="a paper boat drifting down a quiet stream",
        pipeline=pipeline,
        output_type="pt",
        load_type=cast(Any, load_type),
        height=480,
        width=720,
        num_frames=9,
        num_inference_steps=1,
        guidance_scale=1.0,
        seed=42,
    )

    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, 9, 3, 480, 720)
    assert output.isfinite().all()
    assert fps == 8
