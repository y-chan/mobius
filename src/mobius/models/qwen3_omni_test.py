# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Synthetic parity tests for the Qwen3-Omni thinker against HF transformers."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mobius._builder import build_from_module
from mobius._configs import ArchitectureConfig
from mobius._testing.ort_inference import OnnxModelSession
from mobius.integrations.transformers._builder import _select_primary_config
from mobius.models.qwen3_omni import Qwen3OmniThinkerForConditionalGeneration
from mobius.tasks import SpeechLanguageTask

_AUDIO_TOKEN_ID = 100
_NUM_MEL_BINS = 16


def _tiny_hf_model():
    from transformers import Qwen3OmniMoeConfig
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        Qwen3OmniMoeThinkerForConditionalGeneration,
    )

    thinker_config = {
        "audio_token_id": _AUDIO_TOKEN_ID,
        "audio_config": {
            "num_mel_bins": _NUM_MEL_BINS,
            "encoder_layers": 2,
            "encoder_attention_heads": 4,
            "encoder_ffn_dim": 64,
            "d_model": 32,
            "max_source_positions": 64,
            "n_window": 50,
            # One attention block per conv chunk, so the block-diagonal
            # mask is exercised with two chunks of input.
            "n_window_infer": 100,
            "output_dim": 64,
            "downsample_hidden_size": 8,
        },
        "vision_config": {
            "depth": 1,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_heads": 2,
            "out_hidden_size": 64,
            "deepstack_visual_indexes": [0],
        },
        "text_config": {
            "vocab_size": 128,
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 32,
            "moe_intermediate_size": 32,
            "num_experts": 8,
            "num_experts_per_tok": 2,
            "norm_topk_prob": True,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 1_000_000.0,
                "mrope_section": [6, 5, 5],
                "mrope_interleaved": True,
                "interleaved": True,
            },
        },
    }
    config = Qwen3OmniMoeConfig(thinker_config=thinker_config)
    torch.manual_seed(0)
    model = Qwen3OmniMoeThinkerForConditionalGeneration(config.thinker_config).eval()
    return config, model


def _run(model, feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    session = OnnxModelSession(model)
    try:
        return session.run(feeds)
    finally:
        session.close()


@pytest.fixture(scope="module")
def omni():
    hf_config, hf_model = _tiny_hf_model()
    primary, parent, _ = _select_primary_config(hf_config)
    config = ArchitectureConfig.from_transformers(primary, parent_config=parent)
    module = Qwen3OmniThinkerForConditionalGeneration(config)
    package = build_from_module(module, config, task=SpeechLanguageTask())
    # Released checkpoints hold the full Omni model; mimic its key layout.
    state_dict = {f"thinker.{k}": v.detach() for k, v in hf_model.state_dict().items()}
    package.apply_weights(module.preprocess_weights(state_dict))
    return config, hf_model, package


def test_config_resolves_thinker(omni):
    config, _, _ = omni
    assert config.hidden_size == 64
    assert config.num_local_experts == 8
    assert config.attn_qk_norm
    assert config.mrope_interleaved
    assert config.audio.audio_token_id == _AUDIO_TOKEN_ID


@pytest.mark.parametrize("mel_len", [170, 230, 300])
def test_audio_encoder_accepts_unpadded_lengths(omni, mel_len):
    """The Qwen3-Omni processor does not pad mel to a chunk multiple."""
    _, hf_model, package = omni
    rng = np.random.default_rng(mel_len)
    input_features = rng.standard_normal((1, _NUM_MEL_BINS, mel_len)).astype(np.float32)
    feature_attention_mask = np.ones((1, mel_len), dtype=np.int64)

    with torch.no_grad():
        hf_audio = hf_model.get_audio_features(
            torch.from_numpy(input_features),
            torch.from_numpy(feature_attention_mask),
        ).last_hidden_state.numpy()

    audio_out = _run(
        package["audio_encoder"],
        {"input_features": input_features, "feature_attention_mask": feature_attention_mask},
    )
    num_audio_tokens = int(audio_out["audio_feature_lengths"][0])
    assert num_audio_tokens == hf_audio.shape[0]
    np.testing.assert_allclose(
        audio_out["audio_features"][0, :num_audio_tokens], hf_audio, rtol=1e-4, atol=1e-4
    )


def test_three_stage_matches_hf_thinker(omni):
    config, hf_model, package = omni
    rng = np.random.default_rng(0)

    # Two conv chunks, second one partially padded.
    mel_len, valid_mel = 200, 170
    input_features = rng.standard_normal((1, _NUM_MEL_BINS, mel_len)).astype(np.float32)
    feature_attention_mask = np.zeros((1, mel_len), dtype=np.int64)
    feature_attention_mask[:, :valid_mel] = 1

    with torch.no_grad():
        hf_audio = hf_model.get_audio_features(
            torch.from_numpy(input_features),
            torch.from_numpy(feature_attention_mask),
        ).last_hidden_state.numpy()

    audio_out = _run(
        package["audio_encoder"],
        {"input_features": input_features, "feature_attention_mask": feature_attention_mask},
    )
    num_audio_tokens = int(audio_out["audio_feature_lengths"][0])
    assert num_audio_tokens == hf_audio.shape[0]
    audio_features = audio_out["audio_features"][0, :num_audio_tokens]
    np.testing.assert_allclose(audio_features, hf_audio, rtol=1e-4, atol=1e-4)

    input_ids = np.array(
        [[1, 2, *([_AUDIO_TOKEN_ID] * num_audio_tokens), 3, 4]], dtype=np.int64
    )
    inputs_embeds = _run(
        package["embedding"], {"input_ids": input_ids, "audio_features": audio_features}
    )["inputs_embeds"]

    seq_len = input_ids.shape[1]
    attention_mask = np.ones_like(input_ids)
    # Distinct T/H/W positions exercise the interleaved MRoPE layout.
    pos = np.arange(seq_len, dtype=np.int64)
    position_ids = np.stack([pos, pos // 2, pos // 3])[:, None, :]

    with torch.no_grad():
        hf_logits = hf_model(
            input_ids=torch.from_numpy(input_ids),
            input_features=torch.from_numpy(input_features),
            feature_attention_mask=torch.from_numpy(feature_attention_mask),
            attention_mask=torch.from_numpy(attention_mask),
            position_ids=torch.from_numpy(position_ids),
            use_cache=False,
        ).logits.numpy()

    kv_shape = (1, config.num_key_value_heads, 0, config.head_dim)
    decoder_out = _run(
        package["decoder"],
        {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            **{
                f"past_key_values.{i}.{kind}": np.zeros(kv_shape, dtype=np.float32)
                for i in range(config.num_hidden_layers)
                for kind in ("key", "value")
            },
        },
    )
    np.testing.assert_allclose(decoder_out["logits"], hf_logits, rtol=2e-4, atol=2e-4)
