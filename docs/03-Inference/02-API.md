---
---

<!-- TODO: refactor Python API as an unique document, (and redirect related chapters (like Quick_Start.md) to Python API document)-->

# API

<!-- TODO: List all supported oprations in Python API, rather than present as a demo -->

## Python

We provide a Python API for CogKit, including load and inference related operations.

```python
import torch
from PIL import Image

from cogkit import (
    load_pipeline,
    load_lora_checkpoint,
    unload_lora_checkpoint,

    generate_image,
    generate_video,
)
from diffusers.utils import export_to_video


model_id_or_path = "THUDM/CogView4-6B"  # t2i generation task, for example.
pipeline = load_pipeline(
    model_id_or_path,
    transformer_path=None,
    dtype=torch.bfloat16,
)

###### [Optional] Load/Unload LoRA weights
# lora_model_id_or_path = "/path/to/lora/checkpoint"
# load_lora_checkpoint(pipeline, lora_model_id_or_path)
# ...
# unload_lora_checkpoint(pipeline)


###### Text-to-Image generation
batched_image = generate_image(
    prompt="a beautiful sunset over mountains",
    pipeline=pipeline,
    height=1024,
    width=1024,
    output_type="pil",
    load_type="cpu_model_offload",
)
batched_image[0].save("output.png")


###### Text/Image-to-Video generation
batched_video, fps = generate_video(
    prompt="a cat playing with a ball",
    pipeline=pipeline,
    # input_image=Image.open("/path/to/image.png"),  # only for i2v generation
    output_type="pil",
    load_type="cpu_model_offload",
)
export_to_video(batched_video[0], "output.mp4", fps=fps)
```

See function signatures for more details.

`load_type` accepts `cuda`, `mps`, `cpu_model_offload`, or `sequential_cpu_offload`.
Offload modes automatically target the available CUDA or MPS accelerator; direct modes
fail loudly when the requested backend is unavailable. Model CPU offload is the recommended
starting point on Apple Silicon. See the [Apple Silicon guide](../06-Apple-Silicon.md).

<!-- TODO: Add documentation for API server endpoints -->

## API Server Endpoints

The API server reads its placement strategy from `OFFLOAD_TYPE` in `.env`. For example,
the following configuration serves CogView4 using model CPU offload to the detected CUDA
or MPS accelerator:

```bash
DTYPE=bfloat16
OFFLOAD_TYPE=cpu_model_offload
COGVIEW4_PATH=THUDM/CogView4-6B
```

Use `OFFLOAD_TYPE=mps` for explicit full-pipeline MPS placement. When serving on Apple
Silicon, launch the server with `PYTORCH_ENABLE_MPS_FALLBACK=1` if CPU fallback is required.
