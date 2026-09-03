"""Tests for the CogKit-owned CogView4 training attention processor.

The processor exists for speed (build the mixed attention mask once per forward instead of
once per transformer block), so the tests that matter are the ones proving it did not change
the answer. Every case is checked against diffusers' `CogView4TrainingAttnProcessor` running
on the same `Attention` module.
"""

import pytest
import torch
from diffusers.models.attention_processor import Attention
from diffusers.models.transformers.transformer_cogview4 import CogView4TrainingAttnProcessor

from cogkit.finetune.diffusion.models.cogview.cogview4.attention import (
    CogKitCogView4TrainingAttnProcessor,
)

BATCH_SIZE = 2
TEXT_SEQ_LENGTH = 6
IMAGE_SEQ_LENGTH = 10
HEADS = 4
DIM_HEAD = 8
EMBED_DIM = HEADS * DIM_HEAD


def _attention_module() -> Attention:
    torch.manual_seed(0)
    return Attention(query_dim=EMBED_DIM, heads=HEADS, dim_head=DIM_HEAD, bias=True)


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(1)
    encoder_hidden_states = torch.randn(BATCH_SIZE, TEXT_SEQ_LENGTH, EMBED_DIM)
    hidden_states = torch.randn(BATCH_SIZE, IMAGE_SEQ_LENGTH, EMBED_DIM)
    return hidden_states, encoder_hidden_states


def _full_text_mask() -> torch.Tensor:
    return torch.ones((BATCH_SIZE, TEXT_SEQ_LENGTH), dtype=torch.int32)


def _padded_text_mask() -> torch.Tensor:
    mask = torch.ones((BATCH_SIZE, TEXT_SEQ_LENGTH), dtype=torch.int32)
    mask[0, :2] = 0  # left padding, as `process_prompt_attention_mask` produces
    mask[1, :4] = 0
    return mask


def _run_both(text_attn_mask, latent_attn_mask=None, batch_flag=None):
    attn = _attention_module()
    hidden_states, encoder_hidden_states = _inputs()
    upstream = CogView4TrainingAttnProcessor()
    cogkit = CogKitCogView4TrainingAttnProcessor()

    kwargs = dict(
        latent_attn_mask=latent_attn_mask,
        text_attn_mask=text_attn_mask,
        batch_flag=batch_flag,
        image_rotary_emb=None,
    )
    with torch.no_grad():
        expected = upstream(attn, hidden_states, encoder_hidden_states, **kwargs)
        actual = cogkit(attn, hidden_states, encoder_hidden_states, **kwargs)
    return expected, actual, cogkit


@pytest.mark.parametrize(
    "text_attn_mask, latent_attn_mask",
    [
        (None, None),
        (_full_text_mask(), None),
        (_padded_text_mask(), None),
        (_full_text_mask(), torch.ones((BATCH_SIZE, IMAGE_SEQ_LENGTH), dtype=torch.int32)),
        (
            _padded_text_mask(),
            torch.cat(
                [
                    torch.ones((BATCH_SIZE, IMAGE_SEQ_LENGTH - 3), dtype=torch.int32),
                    torch.zeros((BATCH_SIZE, 3), dtype=torch.int32),
                ],
                dim=1,
            ),
        ),
    ],
)
def test_output_matches_upstream(text_attn_mask, latent_attn_mask) -> None:
    expected, actual, _ = _run_both(text_attn_mask, latent_attn_mask)
    for expected_tensor, actual_tensor in zip(expected, actual):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=1e-5, atol=1e-6)


def test_packed_path_delegates_to_upstream() -> None:
    batch_flag = torch.tensor([0, 0], dtype=torch.int32)
    latent_attn_mask = torch.ones((BATCH_SIZE, IMAGE_SEQ_LENGTH), dtype=torch.int32)
    expected, actual, cogkit = _run_both(
        _padded_text_mask(), latent_attn_mask, batch_flag=batch_flag
    )
    for expected_tensor, actual_tensor in zip(expected, actual):
        torch.testing.assert_close(actual_tensor, expected_tensor)
    # The packed path must not populate (or consult) the non-packed mask cache.
    assert cogkit.mask_build_count == 0


def test_all_valid_masks_drop_the_attention_mask() -> None:
    cogkit = CogKitCogView4TrainingAttnProcessor()
    mask = cogkit.mixed_attn_mask(
        text_attn_mask=_full_text_mask(),
        latent_attn_mask=None,
        batch_size=BATCH_SIZE,
        text_seq_length=TEXT_SEQ_LENGTH,
        image_seq_length=IMAGE_SEQ_LENGTH,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    assert mask is None


def test_padded_mask_matches_the_upstream_matrix() -> None:
    text_attn_mask = _padded_text_mask()
    cogkit = CogKitCogView4TrainingAttnProcessor()
    mask = cogkit.mixed_attn_mask(
        text_attn_mask=text_attn_mask,
        latent_attn_mask=None,
        batch_size=BATCH_SIZE,
        text_seq_length=TEXT_SEQ_LENGTH,
        image_seq_length=IMAGE_SEQ_LENGTH,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    # Upstream construction, transcribed from CogView4TrainingAttnProcessor.
    mixed = torch.ones((BATCH_SIZE, TEXT_SEQ_LENGTH + IMAGE_SEQ_LENGTH), dtype=torch.int32)
    mixed[:, :TEXT_SEQ_LENGTH] = text_attn_mask
    mixed_input = mixed.unsqueeze(2).to(dtype=torch.float32)
    expected = (mixed_input @ mixed_input.transpose(1, 2)).to(torch.bool).unsqueeze(1)

    assert mask is not None
    assert torch.equal(mask, expected)


def test_mask_is_built_once_across_blocks() -> None:
    """CogView4-6B has 30 blocks sharing one processor instance; only the first should build."""
    attn = _attention_module()
    hidden_states, encoder_hidden_states = _inputs()
    cogkit = CogKitCogView4TrainingAttnProcessor()
    text_attn_mask = _padded_text_mask()

    with torch.no_grad():
        for _ in range(30):
            cogkit(
                attn,
                hidden_states,
                encoder_hidden_states,
                text_attn_mask=text_attn_mask,
                image_rotary_emb=None,
            )

    assert cogkit.mask_build_count == 1


def test_cache_rebuilds_for_a_new_mask_tensor() -> None:
    cogkit = CogKitCogView4TrainingAttnProcessor()
    common = dict(
        latent_attn_mask=None,
        batch_size=BATCH_SIZE,
        text_seq_length=TEXT_SEQ_LENGTH,
        image_seq_length=IMAGE_SEQ_LENGTH,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    first = _padded_text_mask()
    cogkit.mixed_attn_mask(text_attn_mask=first, **common)
    cogkit.mixed_attn_mask(text_attn_mask=first, **common)
    assert cogkit.mask_build_count == 1

    # A different tensor with different contents must not reuse the cached mask.
    second = _padded_text_mask()
    second[0, 2] = 0
    rebuilt = cogkit.mixed_attn_mask(text_attn_mask=second, **common)
    assert cogkit.mask_build_count == 2
    assert rebuilt is not None
    assert not rebuilt[0, 0, 2, 2]


def test_cache_rebuilds_when_the_mask_is_mutated_in_place() -> None:
    cogkit = CogKitCogView4TrainingAttnProcessor()
    common = dict(
        latent_attn_mask=None,
        batch_size=BATCH_SIZE,
        text_seq_length=TEXT_SEQ_LENGTH,
        image_seq_length=IMAGE_SEQ_LENGTH,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    text_attn_mask = _padded_text_mask()
    cogkit.mixed_attn_mask(text_attn_mask=text_attn_mask, **common)
    text_attn_mask[0, 5] = 0
    cogkit.mixed_attn_mask(text_attn_mask=text_attn_mask, **common)
    assert cogkit.mask_build_count == 2


def test_cache_rebuilds_for_a_new_sequence_length() -> None:
    cogkit = CogKitCogView4TrainingAttnProcessor()
    text_attn_mask = _padded_text_mask()
    common = dict(
        text_attn_mask=text_attn_mask,
        latent_attn_mask=None,
        batch_size=BATCH_SIZE,
        text_seq_length=TEXT_SEQ_LENGTH,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    cogkit.mixed_attn_mask(image_seq_length=IMAGE_SEQ_LENGTH, **common)
    cogkit.mixed_attn_mask(image_seq_length=IMAGE_SEQ_LENGTH * 4, **common)
    assert cogkit.mask_build_count == 2


def test_gradients_match_upstream() -> None:
    """The mask feeds SDPA, so a mask change is a backward change; check it, do not assume it."""
    attn = _attention_module()
    hidden_states, encoder_hidden_states = _inputs()
    text_attn_mask = _padded_text_mask()

    grads = []
    for processor in (CogView4TrainingAttnProcessor(), CogKitCogView4TrainingAttnProcessor()):
        latent = hidden_states.clone().requires_grad_(True)
        text = encoder_hidden_states.clone().requires_grad_(True)
        out_hidden, out_encoder = processor(
            attn, latent, text, text_attn_mask=text_attn_mask, image_rotary_emb=None
        )
        (out_hidden.square().sum() + out_encoder.square().sum()).backward()
        grads.append((latent.grad, text.grad))

    torch.testing.assert_close(grads[1][0], grads[0][0], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(grads[1][1], grads[0][1], rtol=1e-5, atol=1e-6)
