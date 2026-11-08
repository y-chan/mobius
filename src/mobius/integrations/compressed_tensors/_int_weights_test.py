# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for recognising integer weight-only compressed-tensors checkpoints."""

from __future__ import annotations

import copy
import types

import numpy as np
import pytest
import torch

from mobius._configs import QuantizationConfig
from mobius.integrations.compressed_tensors import (
    CompressedTensorsConfig,
    CompressedTensorsError,
    packed_weight_bytes,
    parse_pack_quantized_int_scheme,
)

# The quantization_config llm-compressor writes for W4A16 (here: an AWQ build of
# Qwen3-Omni, trimmed to one ignore entry per kind).
_W4A16 = {
    "config_groups": {
        "group_0": {
            "format": "pack-quantized",
            "input_activations": None,
            "output_activations": None,
            "targets": ["Linear"],
            "weights": {
                "actorder": None,
                "block_structure": None,
                "dynamic": False,
                "group_size": 32,
                "num_bits": 4,
                "observer": "mse",
                "observer_kwargs": {},
                "strategy": "group",
                "symmetric": True,
                "type": "int",
            },
        }
    },
    "format": "pack-quantized",
    "global_compression_ratio": None,
    "ignore": ["thinker.model.layers.0.mlp.gate", "thinker.lm_head"],
    "kv_cache_scheme": None,
    "quant_method": "compressed-tensors",
    "quantization_status": "compressed",
    "sparsity_config": {},
    "transform_config": {},
    "version": "0.11.0",
}


def _variant(**weights) -> dict:
    config = copy.deepcopy(_W4A16)
    config["config_groups"]["group_0"]["weights"].update(weights)
    return config


def test_w4a16_parses_as_integer_affine_quantization():
    scheme = parse_pack_quantized_int_scheme(_W4A16)
    assert scheme is not None
    assert (scheme.num_bits, scheme.group_size) == (4, 32)
    assert scheme.ignore == ("thinker.model.layers.0.mlp.gate", "thinker.lm_head")

    quantization = QuantizationConfig.from_value(_W4A16)
    assert quantization is not None
    assert quantization.quant_method == "compressed-tensors"
    assert (quantization.bits, quantization.group_size, quantization.sym) == (4, 32, True)
    assert quantization.modules_to_not_convert == scheme.ignore


def test_w4a16_is_not_handed_to_the_mixed_format_loader():
    hf_config = types.SimpleNamespace(quantization_config=_W4A16)
    assert CompressedTensorsConfig.from_hf_config(hf_config) is None


def test_other_compressed_tensors_formats_are_left_to_the_mixed_parser():
    mixed = copy.deepcopy(_W4A16)
    mixed["format"] = "mixed-precision"
    mixed["config_groups"]["group_0"]["format"] = "float-quantized"
    assert parse_pack_quantized_int_scheme(mixed) is None
    assert parse_pack_quantized_int_scheme({"quant_method": "gptq", "bits": 4}) is None


@pytest.mark.parametrize(
    ("config", "match"),
    [
        (_variant(symmetric=False), "symmetric"),
        (_variant(num_bits=8), "4-bit"),
        (_variant(strategy="channel", group_size=None), "group-wise"),
        (_variant(dynamic=True), "static"),
        (_variant(group_size=24), "power of two"),
        (_variant(actorder="group"), "actorder"),
    ],
)
def test_unbindable_weight_schemes_fail_closed(config, match):
    with pytest.raises(CompressedTensorsError, match=match):
        parse_pack_quantized_int_scheme(config)


def test_quantized_activations_fail_closed():
    config = copy.deepcopy(_W4A16)
    config["config_groups"]["group_0"]["input_activations"] = {
        "num_bits": 8,
        "type": "int",
        "strategy": "token",
        "symmetric": True,
        "dynamic": True,
    }
    with pytest.raises(CompressedTensorsError, match="weight-only"):
        parse_pack_quantized_int_scheme(config)


def test_packed_bytes_are_the_matmulnbits_nibble_order():
    # compressed-tensors packs value i of a row into bits 4*(i % 8) of word
    # i // 8, offset by 8. MatMulNBits / QMoE read byte i // 2, low nibble
    # first, with an implicit zero point of 8: the same bytes.
    values = torch.tensor([[-8, -1, 0, 7, 3, -4, 5, 1, 2, -2, 6, -7, 0, 0, 1, -1]])
    unsigned = (values + 8).to(torch.int64)
    words = np.zeros((1, 2), dtype=np.uint32)
    for i in range(values.shape[1]):
        words[0, i // 8] |= np.uint32(int(unsigned[0, i]) << (4 * (i % 8)))
    packed = torch.from_numpy(words.view(np.int32))

    data = packed_weight_bytes(packed)
    assert data.dtype == torch.uint8
    assert data.shape == (1, 8)
    low, high = data & 0xF, data >> 4
    nibbles = torch.stack([low, high], dim=-1).reshape(1, -1).to(torch.int64)
    assert torch.equal(nibbles - 8, values)


def test_packed_bytes_reject_non_int32():
    with pytest.raises(CompressedTensorsError, match="int32"):
        packed_weight_bytes(torch.zeros(2, 4, dtype=torch.int64))
