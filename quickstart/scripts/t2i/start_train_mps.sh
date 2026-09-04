#! /usr/bin/env bash
# Single-device Apple Silicon (MPS) launch. Training still goes through torchrun
# (world_size=1, gloo) — see APPLE_METAL_PORT_PLAN.md.

# Ops missing from the MPS backend fall back to CPU instead of crashing
export PYTORCH_ENABLE_MPS_FALLBACK=1
# Force the device explicitly (unset to auto-detect; use `cpu` for parity/oracle runs)
export COGKIT_DEVICE="${COGKIT_DEVICE:-mps}"

# Prefer the repo venv's torchrun so the run sees the editable cogkit install
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
TORCHRUN="${TORCHRUN:-$REPO_ROOT/.venv/bin/torchrun}"
[ -x "$TORCHRUN" ] || TORCHRUN=torchrun

"$TORCHRUN" \
    --nproc_per_node=1 \
    --master_port=29501 \
    ../train.py \
    --yaml "${COGKIT_CONFIG:-config_mps.yaml}"
