# -*- coding: utf-8 -*-
"""CogKit-owned CogView4 training attention processor.

Diffusers' ``CogView4TrainingAttnProcessor`` rebuilds the same mixed text+latent attention
mask inside every transformer block. CogView4-6B has 28 blocks, so one 512x512 forward pays
for 28 identical ``(seq_len, seq_len)`` outer products, and -- because the collated text mask
is never moved off the host -- 28 host-to-device copies. Gradient checkpointing pays for all
of it a second time during backward.

This processor keeps the upstream masking semantics exactly and changes only *when* the mask
is built: once per distinct set of input masks, then reused across every block. When every
token is valid the mask is dropped entirely, because an all-True boolean mask contributes a
zero attention bias and ``attn_mask=None`` is therefore the same computation.

Packed training (``batch_flag`` is not None) is delegated to the upstream processor unchanged.
Its Python loops and scalar materializations need their own profile and parity gate.

Do not expect a speedup from this at 512x512. The 28 rebuilds measure ~25 ms against a
~3.7 s forward, so the win is under half a percent and sits inside run-to-run noise; the
measurement is recorded in APPLE_SILICON_PERFORMANCE_PLAN.md Step 3. What this buys is
correctness-preserving headroom -- the cost grows with sequence length, the mask-drop path
removes a real ~7% SDPA penalty whenever prompts happen to need no padding, and the mask
stops being a per-block host-to-device synchronization point.
"""

import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention
from diffusers.models.transformers.transformer_cogview4 import CogView4TrainingAttnProcessor

__all__ = ["CogKitCogView4TrainingAttnProcessor"]


def _tensor_version(tensor: torch.Tensor | None) -> int:
    return -1 if tensor is None else tensor._version


class _MixedMaskCache:
    """One cached mask plus enough state to prove it still applies.

    Strong references to the source masks are deliberate: ``id()`` is only unique among live
    objects, so holding the tensors is what makes the identity check safe against a freed
    tensor's address being handed to a different one.
    """

    __slots__ = (
        "text_attn_mask",
        "latent_attn_mask",
        "text_version",
        "latent_version",
        "batch_size",
        "text_seq_length",
        "image_seq_length",
        "dtype",
        "device",
        "mask",
    )

    def __init__(
        self,
        text_attn_mask: torch.Tensor | None,
        latent_attn_mask: torch.Tensor | None,
        batch_size: int,
        text_seq_length: int,
        image_seq_length: int,
        dtype: torch.dtype,
        device: torch.device,
        mask: torch.Tensor | None,
    ) -> None:
        self.text_attn_mask = text_attn_mask
        self.latent_attn_mask = latent_attn_mask
        self.text_version = _tensor_version(text_attn_mask)
        self.latent_version = _tensor_version(latent_attn_mask)
        self.batch_size = batch_size
        self.text_seq_length = text_seq_length
        self.image_seq_length = image_seq_length
        self.dtype = dtype
        self.device = device
        self.mask = mask

    def matches(
        self,
        text_attn_mask: torch.Tensor | None,
        latent_attn_mask: torch.Tensor | None,
        batch_size: int,
        text_seq_length: int,
        image_seq_length: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> bool:
        return (
            self.text_attn_mask is text_attn_mask
            and self.latent_attn_mask is latent_attn_mask
            and self.text_version == _tensor_version(text_attn_mask)
            and self.latent_version == _tensor_version(latent_attn_mask)
            and self.batch_size == batch_size
            and self.text_seq_length == text_seq_length
            and self.image_seq_length == image_seq_length
            and self.dtype == dtype
            and self.device == device
        )


class CogKitCogView4TrainingAttnProcessor:
    """Drop-in replacement for ``CogView4TrainingAttnProcessor`` that builds its mask once.

    A single instance is shared by every ``Attention`` module in the transformer (see
    ``replace_attn_processor``), which is what makes the one-entry cache sufficient: the
    blocks run in sequence over the same masks, so the first block builds and the remaining
    27 hit.
    """

    def __init__(self) -> None:
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("CogKitCogView4TrainingAttnProcessor requires PyTorch 2.0 or newer.")
        self._packed_processor = CogView4TrainingAttnProcessor()
        self._cache: _MixedMaskCache | None = None
        # Incremented on every cache miss. Tests assert the reuse; nothing else reads it.
        self.mask_build_count = 0

    def reset_cache(self) -> None:
        self._cache = None

    def mixed_attn_mask(
        self,
        text_attn_mask: torch.Tensor | None,
        latent_attn_mask: torch.Tensor | None,
        batch_size: int,
        text_seq_length: int,
        image_seq_length: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor | None:
        cache = self._cache
        if cache is not None and cache.matches(
            text_attn_mask,
            latent_attn_mask,
            batch_size,
            text_seq_length,
            image_seq_length,
            dtype,
            device,
        ):
            return cache.mask

        mask = self._build_mixed_attn_mask(
            text_attn_mask=text_attn_mask,
            latent_attn_mask=latent_attn_mask,
            batch_size=batch_size,
            text_seq_length=text_seq_length,
            image_seq_length=image_seq_length,
            dtype=dtype,
            device=device,
        )
        self.mask_build_count += 1
        self._cache = _MixedMaskCache(
            text_attn_mask,
            latent_attn_mask,
            batch_size,
            text_seq_length,
            image_seq_length,
            dtype,
            device,
            mask,
        )
        return mask

    def _build_mixed_attn_mask(
        self,
        text_attn_mask: torch.Tensor | None,
        latent_attn_mask: torch.Tensor | None,
        batch_size: int,
        text_seq_length: int,
        image_seq_length: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor | None:
        if text_attn_mask is not None:
            assert text_attn_mask.dim() == 2, (
                "the shape of text_attn_mask should be (batch_size, text_seq_length)"
            )
            assert text_attn_mask.dtype == torch.int32, (
                "the dtype of text_attn_mask should be torch.int32"
            )
        if latent_attn_mask is not None:
            assert latent_attn_mask.dim() == 2, (
                "the shape of latent_attn_mask should be (batch_size, num_latent_tokens)"
            )
            assert latent_attn_mask.dtype == torch.int32, (
                "the dtype of latent_attn_mask should be torch.int32"
            )

        # A missing mask means "every token is valid", which upstream spells as a tensor of
        # ones. Test validity before the device transfer so the readback stays on whichever
        # device already holds the mask -- for the collated text mask that is the host.
        text_all_valid = text_attn_mask is None or bool(text_attn_mask.all())
        latent_all_valid = latent_attn_mask is None or bool(latent_attn_mask.all())
        if text_all_valid and latent_all_valid:
            return None

        mixed_attn_mask = torch.ones(
            (batch_size, text_seq_length + image_seq_length), dtype=torch.int32, device=device
        )
        if text_attn_mask is not None:
            mixed_attn_mask[:, :text_seq_length] = text_attn_mask.to(device=device)
        if latent_attn_mask is not None:
            mixed_attn_mask[:, text_seq_length:] = latent_attn_mask.to(device=device)

        mixed_attn_mask_input = mixed_attn_mask.unsqueeze(2).to(dtype=dtype)
        attn_mask_matrix = mixed_attn_mask_input @ mixed_attn_mask_input.transpose(1, 2)
        # Add the attention head dimension; upstream produces the same bool matrix.
        return attn_mask_matrix.to(dtype=torch.bool).unsqueeze(1)

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        latent_attn_mask: torch.Tensor | None = None,
        text_attn_mask: torch.Tensor | None = None,
        batch_flag: torch.Tensor | None = None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor]
        | list[tuple[torch.Tensor, torch.Tensor]]
        | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if batch_flag is not None:
            # Packed training keeps the upstream path until it has its own profile and gate.
            return self._packed_processor(
                attn,
                hidden_states,
                encoder_hidden_states,
                latent_attn_mask=latent_attn_mask,
                text_attn_mask=text_attn_mask,
                batch_flag=batch_flag,
                image_rotary_emb=image_rotary_emb,
                **kwargs,
            )

        batch_size, text_seq_length, _ = encoder_hidden_states.shape
        _, image_seq_length, _ = hidden_states.shape
        dtype = encoder_hidden_states.dtype
        device = encoder_hidden_states.device

        attention_mask = self.mixed_attn_mask(
            text_attn_mask=text_attn_mask,
            latent_attn_mask=latent_attn_mask,
            batch_size=batch_size,
            text_seq_length=text_seq_length,
            image_seq_length=image_seq_length,
            dtype=dtype,
            device=device,
        )

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        # 1. QKV projections
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        # 2. QK normalization
        if attn.norm_q is not None:
            query = attn.norm_q(query).to(dtype=dtype)
        if attn.norm_k is not None:
            key = attn.norm_k(key).to(dtype=dtype)

        # 3. Rotary positional embeddings, latent stream only
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query[:, :, text_seq_length:, :] = apply_rotary_emb(
                query[:, :, text_seq_length:, :], image_rotary_emb, use_real_unbind_dim=-2
            )
            key[:, :, text_seq_length:, :] = apply_rotary_emb(
                key[:, :, text_seq_length:, :], image_rotary_emb, use_real_unbind_dim=-2
            )

        # 4. Attention
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        # 5. Output projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
        )
        return hidden_states, encoder_hidden_states
