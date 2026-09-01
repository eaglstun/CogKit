from typing import Literal, TypeAlias

LoadType: TypeAlias = Literal[
    "cuda",
    "mps",
    "cpu_model_offload",
    "sequential_cpu_offload",
]
