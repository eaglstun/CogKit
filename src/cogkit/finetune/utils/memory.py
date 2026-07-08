import gc
from typing import Any

import torch


def get_memory_statistics(device: torch.device, precision: int = 3) -> dict[str, Any]:
    memory_allocated = None
    memory_reserved = None
    max_memory_allocated = None
    max_memory_reserved = None

    match device.type:
        case "cuda":
            device = torch.cuda.current_device()
            memory_allocated = torch.cuda.memory_allocated(device)
            memory_reserved = torch.cuda.memory_reserved(device)
            max_memory_allocated = torch.cuda.max_memory_allocated(device)
            max_memory_reserved = torch.cuda.max_memory_reserved(device)
        case "mps":
            # MPS only exposes current/driver allocation; no reserved or peak stats
            memory_allocated = torch.mps.current_allocated_memory()
            memory_reserved = torch.mps.driver_allocated_memory()
        case _:
            pass

    return {
        "memory_allocated": round_to_gigabytes(memory_allocated, precision),
        "memory_reserved": round_to_gigabytes(memory_reserved, precision),
        "max_memory_allocated": round_to_gigabytes(max_memory_allocated, precision),
        "max_memory_reserved": round_to_gigabytes(max_memory_reserved, precision),
    }


def round_to_gigabytes(x: int | None, precision: int = 3) -> float | None:
    if x is None:
        return None
    return round(x / 1024**3, ndigits=precision)


def bytes_to_gigabytes(x: int) -> float:
    if x is not None:
        return x / 1024**3


def free_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
