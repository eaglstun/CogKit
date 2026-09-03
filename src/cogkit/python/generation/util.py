# -*- coding: utf-8 -*-

import torch
from torch import nn
from diffusers import (
    CogVideoXDPMScheduler,
    CogVideoXImageToVideoPipeline,
    CogVideoXPipeline,
    CogView4ControlPipeline,
    CogView4Pipeline,
)

from cogkit.logging import get_logger
from cogkit.types import LoadType
from cogkit.utils import get_device

TVideoPipeline = CogVideoXPipeline | CogVideoXImageToVideoPipeline
TPipeline = CogView4Pipeline | TVideoPipeline
CogviewPipline = CogView4Pipeline | CogView4ControlPipeline
_COGKIT_LOAD_TYPE_ATTR = "_cogkit_load_type"
_logger = get_logger(__name__)


def _convert_mps_float64_buffers(pipeline: TPipeline) -> list[str]:
    """Convert registered buffers that MPS cannot represent, leaving parameters untouched."""
    converted = []
    for component_name, component in getattr(pipeline, "components", {}).items():
        if not isinstance(component, nn.Module):
            continue
        for module_name, module in component.named_modules():
            for buffer_name, buffer in module.named_buffers(recurse=False):
                if buffer.dtype != torch.float64:
                    continue
                setattr(module, buffer_name, buffer.float())
                qualified_name = ".".join(
                    part for part in (component_name, module_name, buffer_name) if part
                )
                converted.append(qualified_name)
    return converted


def _is_cogvideox1_0(pipeline: TVideoPipeline) -> bool:
    # ! very hacky
    if isinstance(pipeline, CogVideoXPipeline | CogVideoXImageToVideoPipeline):
        return (
            not hasattr(pipeline.transformer.config, "patch_size_t")
            or pipeline.transformer.config.patch_size_t is None
        )

    raise ValueError(
        f"Unsupported pipeline type in `_is_cogvideox1_0`, pipeline type: {type(pipeline)}"
    )


def _is_cogvideox1_5(pipeline: TVideoPipeline) -> bool:
    # ! very hacky
    if isinstance(pipeline, CogVideoXPipeline | CogVideoXImageToVideoPipeline):
        return (
            hasattr(pipeline.transformer.config, "patch_size_t")
            and pipeline.transformer.config.patch_size_t == 2
        )
    raise ValueError(
        f"Unsupported pipeline type in `_is_cogvideox1_5`, pipeline type: {type(pipeline)}"
    )


def _guess_cogview_resolution(
    pipeline: CogView4Pipeline, height: int | None = None, width: int | None = None
) -> tuple[int, int]:
    default_height = pipeline.transformer.config.sample_size * pipeline.vae_scale_factor
    default_width = pipeline.transformer.config.sample_size * pipeline.vae_scale_factor
    if height is None and width is None:
        return default_height, default_width

    if height is None:
        height = int(width * default_height / default_width)

    if width is None:
        width = int(height * default_width / default_height)

    # * Check resolution according to the model card
    assert height is not None and width is not None
    if isinstance(pipeline, CogView4Pipeline):
        assert height % 32 == 0 and width % 32 == 0, "height and width must be divisible by 32"
        return height, width

    raise ValueError(
        f"Unsupported pipeline type in `_guess_cogview_resolution`, pipeline type: {type(pipeline)}"
    )


def _guess_cogvideox_resolution(
    pipeline: TVideoPipeline, height: int | None, width: int | None = None
) -> tuple[int, int]:
    default_height = pipeline.transformer.config.sample_height * pipeline.vae_scale_factor_spatial
    default_width = pipeline.transformer.config.sample_width * pipeline.vae_scale_factor_spatial

    if height is None and width is None:
        height, width = default_height, default_width
    elif height is None:
        height = int(width * default_height / default_width)
    elif width is None:
        width = int(height * default_width / default_height)

    # * Check resolution according to the model card
    if _is_cogvideox1_0(pipeline):
        assert height == 480 and width == 720, "height and width must be 480 and 720"
    elif _is_cogvideox1_5(pipeline):
        if isinstance(pipeline, CogVideoXPipeline):
            assert height == 768 and width == 1360, "height and width must be 768 and 1360"
        elif isinstance(pipeline, CogVideoXImageToVideoPipeline):
            minv = min(height, width)
            maxv = max(height, width)
            assert minv == 768, "minimum value in (height, width) must be 768"
            assert 768 <= maxv <= 1360, (
                "maximum value in (height, width) must range from 768 to 1360"
            )
            assert maxv % 16 == 0, "maximum value in (height, width) must be divisible by 16"
    else:
        raise ValueError(
            f"Unsupported pipeline type in `_guess_cogvideox_resolution`, pipeline type: {type(pipeline)}"
        )

    return height, width


def guess_resolution(
    pipeline: TPipeline,
    height: int | None = None,
    width: int | None = None,
) -> tuple[int, int]:
    if isinstance(pipeline, CogviewPipline):
        return _guess_cogview_resolution(pipeline, height=height, width=width)
    if isinstance(pipeline, TVideoPipeline):
        return _guess_cogvideox_resolution(pipeline, height=height, width=width)

    err_msg = f"The pipeline '{pipeline.__class__.__name__}' is not supported."
    raise ValueError(err_msg)


def guess_frames(pipeline: TVideoPipeline, frames: int | None = None) -> tuple[int, int]:
    if frames is None:
        frames = pipeline.transformer.config.sample_frames

    ##### Check frames according to model card
    if _is_cogvideox1_0(pipeline):
        assert frames <= 49, "frames must <=49"
        assert (frames - 1) % 8 == 0, "frames must be 8N+1"
        fps = 8
    elif _is_cogvideox1_5(pipeline):
        assert frames <= 81, "frames must <=81"
        assert (frames - 1) % 16 == 0, "frames must be 16N+1"
        fps = 16
    else:
        raise ValueError(
            f"Unsupported pipeline type in `guess_frames`, pipeline type: {type(pipeline)}"
        )

    return frames, fps


def before_generation(
    pipeline: TPipeline,
    load_type: LoadType = "cpu_model_offload",
) -> None:
    if isinstance(pipeline, TVideoPipeline) and not isinstance(
        pipeline.scheduler, CogVideoXDPMScheduler
    ):
        pipeline.scheduler = CogVideoXDPMScheduler.from_config(
            pipeline.scheduler.config, timestep_spacing="trailing"
        )

    if getattr(pipeline, _COGKIT_LOAD_TYPE_ATTR, None) == load_type:
        return

    if load_type in ("cuda", "mps"):
        if load_type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA inference was requested, but CUDA is not available")
        if load_type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS inference was requested, but MPS is not available")
        if load_type == "mps":
            converted = _convert_mps_float64_buffers(pipeline)
            if converted:
                _logger.info("Converted MPS-incompatible float64 buffers: %s", converted)
        pipeline.remove_all_hooks()
        pipeline.to(load_type)
    elif load_type == "cpu_model_offload":
        device = get_device()
        if device.type == "cpu":
            raise RuntimeError("CPU model offload requires a CUDA or MPS accelerator")
        if device.type == "mps":
            converted = _convert_mps_float64_buffers(pipeline)
            if converted:
                _logger.info("Converted MPS-incompatible float64 buffers: %s", converted)
        pipeline.enable_model_cpu_offload(device=device)
    elif load_type == "sequential_cpu_offload":
        device = get_device()
        if device.type == "cpu":
            raise RuntimeError("Sequential CPU offload requires a CUDA or MPS accelerator")
        if device.type == "mps":
            converted = _convert_mps_float64_buffers(pipeline)
            if converted:
                _logger.info("Converted MPS-incompatible float64 buffers: %s", converted)
        pipeline.enable_sequential_cpu_offload(device=device)
    else:
        raise ValueError(f"Unsupported offload type: {load_type}")
    if hasattr(pipeline, "vae"):
        pipeline.vae.enable_slicing()
        pipeline.vae.enable_tiling()
    setattr(pipeline, _COGKIT_LOAD_TYPE_ATTR, load_type)
