# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Synthetic parity tests for the Qwen3-Omni thinker against HF transformers."""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
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


# ---------------------------------------------------------------------------
# Expert-only int8 checkpoints (QMoE)
# ---------------------------------------------------------------------------

_INT8_BLOCK = 32


def _quantize_int8_block(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric int8 with one fp32 absmax scale per ``_INT8_BLOCK`` inputs.

    Blocks run along the last (input) dimension, as in SoulX-Transcriber's
    ``make_int8_base32.py``. Returns ``(int8 [..., out, in], scale [..., out, n_blocks])``.
    """
    blocks = weight.float().reshape(-1, _INT8_BLOCK)
    scale = blocks.abs().amax(-1, keepdim=True).clamp_min(1e-30) / 127.0
    q = torch.round(blocks / scale).clamp_(-127, 127).to(torch.int8).reshape(weight.shape)
    return q, scale.reshape(*weight.shape[:-1], -1)


def _dequantize_int8_block(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (q.float().reshape(*scale.shape, _INT8_BLOCK) * scale.unsqueeze(-1)).reshape(
        q.shape
    )


def _int8_expert_checkpoint(hf_model, num_layers: int) -> tuple[dict, dict]:
    """Olive-layout state dict with int8 experts; the HF model gets the dequantized experts.

    Returns ``(state_dict, quantization_config)``. Mutating the HF experts to
    their dequantized values makes it the exact float reference for the
    QMoE graph, so any mismatch comes from the export, not from rounding.
    """
    quantization_config = {
        "quant_method": "olive",
        "bits": 8,
        "group_size": _INT8_BLOCK,
        "sym": True,
        "modules_to_not_convert": [
            f"thinker.model.layers.{i}.self_attn" for i in range(num_layers)
        ],
    }
    state_dict = {}
    with torch.no_grad():
        for i, layer in enumerate(hf_model.model.layers):
            for name in ("gate_up_proj", "down_proj"):
                param = getattr(layer.mlp.experts, name)
                q, scale = _quantize_int8_block(param)
                param.copy_(_dequantize_int8_block(q, scale))
                prefix = f"thinker.model.layers.{i}.mlp.experts.{name}"
                # Olive stores symmetric int8 as uint8 with an implicit zero point of 128.
                state_dict[f"{prefix}_qweight"] = (q.to(torch.int16) + 128).to(torch.uint8)
                state_dict[f"{prefix}_scales"] = scale
        for key, value in hf_model.state_dict().items():
            if ".mlp.experts." not in key:
                state_dict[f"thinker.{key}"] = value.detach()
    return state_dict, quantization_config


@pytest.fixture(scope="module")
def omni_int8():
    hf_config, hf_model = _tiny_hf_model()
    num_layers = hf_config.thinker_config.text_config.num_hidden_layers
    state_dict, quantization_config = _int8_expert_checkpoint(hf_model, num_layers)
    hf_config.quantization_config = quantization_config
    primary, parent, _ = _select_primary_config(hf_config)
    config = ArchitectureConfig.from_transformers(primary, parent_config=parent)
    module = Qwen3OmniThinkerForConditionalGeneration(config)
    package = build_from_module(module, config, task=SpeechLanguageTask())
    package.apply_weights(module.preprocess_weights(state_dict))
    return config, hf_model, package


def test_int8_experts_use_qmoe_with_float_attention(omni_int8):
    config, _, package = omni_int8
    assert config.quantization.bits == 8
    graph = package["decoder"].graph
    qmoe = [n for n in graph.all_nodes() if n.op_type == "QMoE"]
    assert len(qmoe) == config.num_hidden_layers
    assert {n.attributes["expert_weight_bits"].value for n in qmoe} == {8}
    assert {n.attributes["block_size"].value for n in qmoe} == {_INT8_BLOCK}
    # Attention stays float: no MatMulNBits anywhere in the decoder.
    assert not [n for n in graph.all_nodes() if n.op_type == "MatMulNBits"]
    assert (
        graph.initializers["decoder.layers.0.mlp.fc1_experts_weights"].dtype
        == ir.DataType.UINT8
    )


def test_int8_experts_match_dequantized_hf(omni_int8):
    config, hf_model, package = omni_int8
    rng = np.random.default_rng(1)
    mel_len = 200
    input_features = rng.standard_normal((1, _NUM_MEL_BINS, mel_len)).astype(np.float32)
    feature_attention_mask = np.ones((1, mel_len), dtype=np.int64)
    audio_out = _run(
        package["audio_encoder"],
        {"input_features": input_features, "feature_attention_mask": feature_attention_mask},
    )
    num_audio_tokens = int(audio_out["audio_feature_lengths"][0])
    audio_features = audio_out["audio_features"][0, :num_audio_tokens]
    input_ids = np.array(
        [[1, 2, *([_AUDIO_TOKEN_ID] * num_audio_tokens), 3, 4]], dtype=np.int64
    )
    inputs_embeds = _run(
        package["embedding"], {"input_ids": input_ids, "audio_features": audio_features}
    )["inputs_embeds"]
    seq_len = input_ids.shape[1]
    attention_mask = np.ones_like(input_ids)
    pos = np.arange(seq_len, dtype=np.int64)
    position_ids = np.stack([pos, pos, pos])[:, None, :]

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
    # Correct layouts land ~1e-7 apart; a gate/up row swap is ~5e-3 and a 5%
    # scale error ~1e-3, so this bound separates them with margin.
    np.testing.assert_allclose(decoder_out["logits"], hf_logits, rtol=1e-5, atol=1e-5)


def test_partially_float_attention_is_rejected():
    hf_config, _ = _tiny_hf_model()
    hf_config.quantization_config = {
        "quant_method": "olive",
        "bits": 8,
        "group_size": _INT8_BLOCK,
        "sym": True,
        "modules_to_not_convert": ["thinker.model.layers.0.self_attn"],
    }
    primary, parent, _ = _select_primary_config(hf_config)
    config = ArchitectureConfig.from_transformers(primary, parent_config=parent)
    with pytest.raises(ValueError, match="every layer"):
        Qwen3OmniThinkerForConditionalGeneration(config)


# ---------------------------------------------------------------------------
# Static KV cache decoder (fixed shapes for CUDA graph capture)
# ---------------------------------------------------------------------------

_STATIC_MAX_SEQ_LEN = 64


def test_static_cache_decoder_matches_dynamic(omni):
    """Prefill + decode through pre-allocated KV buffers equals the growing-cache graph."""
    config, hf_model, dynamic_package = omni
    module = Qwen3OmniThinkerForConditionalGeneration(config)
    static_package = build_from_module(
        module,
        config,
        task=SpeechLanguageTask(static_cache=True, max_seq_len=_STATIC_MAX_SEQ_LEN),
    )
    state_dict = {f"thinker.{k}": v.detach() for k, v in hf_model.state_dict().items()}
    static_package.apply_weights(module.preprocess_weights(state_dict))

    static_inputs = {i.name for i in static_package["decoder"].graph.inputs}
    assert "attention_mask" not in static_inputs
    assert {"write_indices", "nonpad_kv_seqlen", "key_cache.0"} <= static_inputs

    rng = np.random.default_rng(3)
    prompt = rng.standard_normal((1, 6, config.hidden_size)).astype(np.float32)
    steps = [prompt] + [
        rng.standard_normal((1, 1, config.hidden_size)).astype(np.float32) for _ in range(4)
    ]
    layers = range(config.num_hidden_layers)
    kv_hidden = config.num_key_value_heads * config.head_dim
    dynamic_past = {
        f"past_key_values.{i}.{kind}": np.zeros(
            (1, config.num_key_value_heads, 0, config.head_dim), np.float32
        )
        for i in layers
        for kind in ("key", "value")
    }
    static_cache = {
        f"{kind}_cache.{i}": np.zeros((1, _STATIC_MAX_SEQ_LEN, kv_hidden), np.float32)
        for i in layers
        for kind in ("key", "value")
    }
    past_len = 0
    for embeds in steps:
        seq = embeds.shape[1]
        pos = np.arange(past_len, past_len + seq, dtype=np.int64)
        position_ids = np.stack([pos, pos, pos])[:, None, :]
        dynamic_out = _run(
            dynamic_package["decoder"],
            {
                "inputs_embeds": embeds,
                "attention_mask": np.ones((1, past_len + seq), np.int64),
                "position_ids": position_ids,
                **dynamic_past,
            },
        )
        static_out = _run(
            static_package["decoder"],
            {
                "inputs_embeds": embeds,
                "position_ids": position_ids,
                "write_indices": np.array([past_len], np.int64),
                "nonpad_kv_seqlen": np.array([past_len + seq], np.int64),
                **static_cache,
            },
        )
        np.testing.assert_allclose(
            static_out["logits"], dynamic_out["logits"], rtol=1e-5, atol=1e-5
        )
        dynamic_past = {
            name: dynamic_out[name.replace("past_key_values.", "present.")]
            for name in dynamic_past
        }
        static_cache = {name: static_out[f"updated_{name}"] for name in static_cache}
        past_len += seq
