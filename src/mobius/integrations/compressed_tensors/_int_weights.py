# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Integer weight-only compressed-tensors checkpoints (``pack-quantized``).

llm-compressor's weight-only integer schemes (W4A16 from GPTQ, AWQ or RTN)
store each quantized ``Linear`` as three tensors:

- ``weight_packed``: int32 ``[out, in * num_bits / 32]``. Each word holds
  ``32 / num_bits`` values, the first in the least significant bits, each
  stored as ``q + 2**(num_bits - 1)`` so the stored nibble is unsigned.
- ``weight_scale``: ``[out, in / group_size]``, one scale per group of input
  features.
- ``weight_shape``: int64 ``[2]``, the logical ``[out, in]``.

For a symmetric scheme the dequantization is ``(nibble - 8) * scale``. That
is what ``com.microsoft::MatMulNBits`` and ``com.microsoft::QMoE`` compute
with their implicit 4-bit zero point of 8, and the little-endian bytes of
``weight_packed`` are those kernels' packed bytes, so binding them is a
reinterpretation rather than a requantization.

This is a different container from the 0.17.2 mixed FP8/NVFP4 format the
rest of this package loads. It is recognised by its own config shape, parsed
into an integer-affine :class:`~mobius._configs.QuantizationConfig`, and bound
by models that implement a streaming plan for it.
"""

from __future__ import annotations

import dataclasses
import sys
from collections.abc import Mapping

import torch

from mobius.integrations.compressed_tensors._loader import (
    CompressedTensorsError,
    QuantizationArgs,
    _normalize_config_value,
    is_compressed_tensors_config,
)

PACK_QUANTIZED_FORMAT = "pack-quantized"


@dataclasses.dataclass(frozen=True)
class PackQuantizedIntScheme:
    """A validated weight-only integer ``pack-quantized`` configuration."""

    num_bits: int
    group_size: int
    ignore: tuple[str, ...]


def parse_pack_quantized_int_scheme(value: object) -> PackQuantizedIntScheme | None:
    """Return the scheme when *value* is an integer weight-only pack-quantized config.

    Returns ``None`` for anything that is not compressed-tensors or does not
    declare ``pack-quantized``; those stay with the mixed-format parser. Once a
    config does declare it, every deviation from static, symmetric, group-wise
    INT4 weights with float activations raises: the packed-byte
    reinterpretation is exact only for that scheme.
    """
    value = _normalize_config_value(value)
    if not is_compressed_tensors_config(value):
        return None
    assert isinstance(value, Mapping)
    groups = value.get("config_groups")
    if not isinstance(groups, Mapping) or not groups:
        return None
    formats = {value.get("format")}
    for group in groups.values():
        if isinstance(group, Mapping):
            formats.add(group.get("format"))
    formats.discard(None)
    if PACK_QUANTIZED_FORMAT not in formats:
        return None
    if formats != {PACK_QUANTIZED_FORMAT}:
        raise CompressedTensorsError(
            f"pack-quantized cannot be mixed with other formats; got {sorted(formats)}."
        )
    if len(groups) != 1:
        raise CompressedTensorsError(
            f"Only one pack-quantized config group is supported; got {sorted(groups)}."
        )
    ((name, group),) = groups.items()
    where = f"config_groups.{name}"
    if not isinstance(group, Mapping):
        raise CompressedTensorsError(f"{where} must be an object.")
    targets = group.get("targets")
    if targets != ["Linear"]:
        raise CompressedTensorsError(f"{where}.targets must be ['Linear']; got {targets!r}.")
    for field in ("input_activations", "output_activations"):
        if group.get(field) is not None:
            raise CompressedTensorsError(
                f"{where}.{field} is set; only weight-only pack-quantized "
                "checkpoints (float activations) are supported."
            )
    for field in ("kv_cache_scheme", "sparsity_config", "transform_config"):
        if value.get(field):
            raise CompressedTensorsError(
                f"quantization_config.{field} is set; pack-quantized checkpoints "
                "carrying it are not supported."
            )
    status = value.get("quantization_status")
    if status not in (None, "compressed"):
        raise CompressedTensorsError(
            f"quantization_status must be 'compressed'; got {status!r}."
        )
    raw_weights = group.get("weights")
    weights = QuantizationArgs.parse(raw_weights, where=f"{where}.weights")
    assert isinstance(raw_weights, Mapping)
    if (
        weights.type != "int"
        or weights.num_bits != 4
        or weights.strategy != "group"
        or not weights.symmetric
        or weights.dynamic is not False
    ):
        raise CompressedTensorsError(
            f"{where}.weights must be static symmetric group-wise 4-bit integers; got "
            f"type={weights.type!r} num_bits={weights.num_bits} "
            f"strategy={weights.strategy!r} symmetric={weights.symmetric} "
            f"dynamic={weights.dynamic!r}."
        )
    group_size = weights.group_size
    if group_size is None or group_size < 16 or group_size & (group_size - 1):
        raise CompressedTensorsError(
            f"{where}.weights.group_size must be a power of two >= 16 (the QMoE "
            f"block-size contract); got {group_size!r}."
        )
    for field in ("actorder", "block_structure"):
        if raw_weights.get(field) is not None:
            raise CompressedTensorsError(
                f"{where}.weights.{field} is {raw_weights[field]!r}; only contiguous "
                "groups without reordering are supported."
            )
    ignore = value.get("ignore") or []
    if not isinstance(ignore, list) or not all(isinstance(n, str) for n in ignore):
        raise CompressedTensorsError("quantization_config.ignore must be a list of names.")
    return PackQuantizedIntScheme(num_bits=4, group_size=group_size, ignore=tuple(ignore))


def packed_weight_bytes(packed: torch.Tensor) -> torch.Tensor:
    """View an int32 ``weight_packed`` ``[out, words]`` as uint8 ``[out, 4 * words]``.

    Read little-endian, byte ``i // 2`` holds value ``i`` in its low nibble for
    even ``i``: the MatMulNBits / QMoE 4-bit packing.
    """
    if packed.dtype != torch.int32 or packed.dim() != 2:
        raise CompressedTensorsError(
            "weight_packed must be a rank-2 int32 tensor; got "
            f"{packed.dtype} {list(packed.shape)}."
        )
    if sys.byteorder != "little":
        raise CompressedTensorsError(
            "Reinterpreting weight_packed needs a little-endian host."
        )
    return packed.contiguous().view(torch.uint8)
