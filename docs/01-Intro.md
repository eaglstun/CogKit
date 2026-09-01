---
slug: /
---

# Introduction

CogKit is an open-source project that provides a user-friendly interface for researchers and developers to utilize ZhipuAI's [CogView](https://huggingface.co/collections/THUDM/cogview-67ac3f241eefad2af015669b) (image generation) and [CogVideoX](https://huggingface.co/collections/THUDM/cogvideo-66c08e62f1685a3ade464cce) (video generation) models. It streamlines multimodal tasks such as text-to-image(T2I), text-to-video(T2V), and image-to-video(I2V). Users must comply with legal and ethical guidelines to ensure responsible implementation.

## Supported Models

Please refer to the [Model Card](./05-Model%20Card.mdx) for more details.

## Environment Testing

The upstream CUDA lane has been tested with 8×A100 GPUs, CUDA 12.4, and Python 3.10.16.
This fork also has an Apple Silicon lane tested on an M4 Max with 64 GB unified memory and
Python 3.12. See the [Apple Silicon guide](./06-Apple-Silicon.md) for its narrower support
matrix.
