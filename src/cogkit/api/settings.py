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

    # cogview-4 related settings
    cogview4_path: str | None = None
    cogview4_transformer_path: str | None = None
    lora_dir: str | None = None

    # prompt generation related settings
    openai_api_key: str | None = None
    openai_base_url: str | None = None
