# -*- coding: utf-8 -*-


from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal

from cogkit.types import LoadType


class APISettings(BaseSettings):
    model_config = SettingsConfigDict(
        extra="ignore", validate_default=True, validate_assignment=True, env_file=".env"
    )
    _supported_models: tuple[str, ...] = ("cogview-4",)

    dtype: Literal["bfloat16", "float32"] = "bfloat16"
    offload_type: LoadType = "cpu_model_offload"
    # VAE slicing + tiling. Defaults on because the server's memory budget is unknown;
    # turn it off only on a host with known headroom. See
    # docs/benchmarks/APPLE_MPS_VAE_DECODE_2026-09-03.md
    vae_memory_saving: bool = True

    # cogview-4 related settings
    cogview4_path: str | None = None
    cogview4_transformer_path: str | None = None
    lora_dir: str | None = None

    # prompt generation related settings
    openai_api_key: str | None = None
    openai_base_url: str | None = None
