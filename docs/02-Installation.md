---
---

# Installation

## Requirements

- Python 3.10 or higher
- PyTorch, OpenCV

## Installation Steps

### PyTorch

Please refer to the [PyTorch installation guide](https://pytorch.org/get-started/locally/) for instructions on installing PyTorch according to your system.

### OpenCV

Please refer to the [OpenCV installation guide](https://github.com/opencv/opencv-python?tab=readme-ov-file#installation-and-usage) to install opencv-python. In most cases, you can simply install by `pip install opencv-python-headless`

### CogKit

Install `cogkit` from github source:

```bash
pip install "cogkit@git+https://github.com/THUDM/cogkit.git"
```

### Apple Silicon

The Apple Silicon lane in this fork is tested with Python 3.12 in a `uv` virtual
environment. Install the local checkout when using its MPS support:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch torchvision
uv pip install --python .venv/bin/python -e ".[finetune]" pytest
```

See the [Apple Silicon guide](./06-Apple-Silicon.md) for supported features, inference
placement choices, and the training recipe.

### Verify installation

You can verify that cogkit is installed correctly by running:

```bash
cogkit --help
```
