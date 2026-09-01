import os

import torch


def get_device(local_rank: int | None = None) -> torch.device:
    override = os.environ.get("COGKIT_DEVICE")
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda" if local_rank is None else f"cuda:{local_rank}")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
