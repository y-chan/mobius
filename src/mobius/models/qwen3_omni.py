# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Qwen3-Omni thinker: audio + text → text with a sparse MoE decoder.

Architecture:
  - Audio encoder: identical to Qwen3-ASR (3x Conv2d downsampling →
    sinusoidal PE → N bidirectional encoder layers → LayerNorm →
    proj1 → GELU → proj2), reused from :mod:`mobius.models.qwen3_asr`
  - Text decoder: Qwen3-MoE layers (QK norm, softmax top-k routing, no
    shared expert) with interleaved MRoPE
  - Fusion: audio features replace ``audio_token_id`` positions in the
    text embeddings

Only the thinker is exported. The vision tower (deepstack), the talker
and code2wav are out of scope; their weights are dropped.

Decoding this export at speed takes an execution provider, session options
and a driving loop that the module itself says nothing about -- the graph is
correct without them, only several times slower, and two of the settings
corrupt the output rather than failing. See
``docs/design/qwen3-omni-cuda-graph-decoding.md``.

Reference: https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct
HuggingFace class: Qwen3OmniMoeThinkerForConditionalGeneration
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import TYPE_CHECKING, ClassVar

import onnx_ir as ir
import torch
from onnxscript import OpBuilder, nn

from mobius._build_context import ep_capabilities
from mobius._configs import ArchitectureConfig, QuantizedWeightFormat
from mobius._weight_utils import preprocess_quantized_weights, quantize_linear_int8_blocks
from mobius.components import (
    FusedQKVAttention,
    Linear,
    MoELayer,
    RMSNorm,
    SoftmaxTopKGate,
    TransposedLinear,
    initialize_rope,
    make_quantized_linear_factory,
)
from mobius.components._moe import _realize_gate_and_get_qmoe_routing
from mobius.models.moe import (
    MoEDecoderLayer,
    _quantized_linear_class,
    _rename_moe_expert_weights,
)

if TYPE_CHECKING:
    from mobius.integrations._weight_loading import StreamingWeightPlan
from mobius.models.qwen3_asr import (
    Qwen3ASRAudioEncoder,
    Qwen3ASRDecoderModel,
    Qwen3ASREmbeddingModel,
)

# Checkpoint prefixes of components outside the thinker's audio + text path.
_DROPPED_PREFIXES = ("talker.", "code2wav.")
_DROPPED_THINKER_PREFIXES = ("visual.",)


def _is_quantized(config: ArchitectureConfig) -> bool:
    quantization = getattr(config, "quantization", None)
    return quantization is not None and quantization.quant_method != "none"


def _is_pack_quantized(config: ArchitectureConfig) -> bool:
    """Whether the checkpoint is compressed-tensors weight-only ``pack-quantized`` INT4.

    llm-compressor's W4A16 output (GPTQ / AWQ / RTN). Its packed words are
    bound by :meth:`Qwen3OmniThinkerForConditionalGeneration.build_pack_quantized_streaming_plan`.
    """
    return _is_quantized(config) and config.quantization.quant_method == "compressed-tensors"


def _float_attention_in_quantized_checkpoint(config: ArchitectureConfig) -> bool:
    """Whether a quantized checkpoint keeps every layer's attention in float.

    Expert-only checkpoints (e.g. int8 routed experts with bf16 everything
    else) declare the float modules in ``quantization_config.modules_to_not_convert``
    using HF root-relative names (``thinker.model.layers.{i}.self_attn``). The
    routed experts then go through QMoE while attention keeps plain ``Linear``
    projections. Checkpoints that list nothing quantize attention as well.
    Mixed per-layer plans are rejected rather than guessed.
    """
    names = config.quantization.modules_to_not_convert or ()
    if any(name.startswith("re:") for name in names):
        raise NotImplementedError(
            "Qwen3-Omni does not resolve regex modules_to_not_convert rules; "
            "list the float modules explicitly."
        )
    names = tuple(name.removeprefix("thinker.") for name in names)

    def excluded(path: str) -> bool:
        return any(path == name or path.startswith(f"{name}.") for name in names)

    layers = range(config.num_hidden_layers)
    if any(excluded(f"model.layers.{i}.mlp.experts") for i in layers):
        raise ValueError(
            "Qwen3-Omni quantized checkpoints must quantize the routed experts; "
            "modules_to_not_convert excludes them."
        )
    float_attention = [excluded(f"model.layers.{i}.self_attn") for i in layers]
    if all(float_attention):
        return True
    if any(float_attention):
        raise ValueError(
            "Qwen3-Omni quantized checkpoints must keep attention either float in "
            "every layer or quantized in every layer."
        )
    return False


def _use_fused_moe(config: ArchitectureConfig) -> bool:
    """Whether to emit one ``com.microsoft::MoE`` node per layer.

    The loop-over-experts fallback materialises ``3 * num_local_experts``
    initializers and as many GEMMs per layer (18k tensors / 87k nodes for the
    30B checkpoint), which dominates session initialisation. Quantized
    checkpoints keep the QMoE path in :class:`~mobius.components.MoELayer`.
    """
    if _is_quantized(config):
        return False
    return ep_capabilities().supports_fused_moe


def _packs_quantized_qkv(config: ArchitectureConfig) -> bool:
    """Whether checkpoint-quantized attention joins Q/K/V into one ``MatMulNBits``.

    ``MatMulNBits`` weights, scales and zero points are all row-major in the
    output features, so three projections join by concatenating rows -- exact,
    nothing requantized -- when they share one integer-affine scheme: the
    model-wide bits, block and zero-point setting with no per-module override.
    That is the case for GPTQ / AWQ / Olive / compressed-tensors checkpoints of
    at most 8 bits.

    One ``N = 5120`` GEMV instead of 4096 + 512 + 512 removes the two small calls,
    which cost the most per byte. On an RTX 3090 at decode, the INT4 Q/K/V of a
    layer go from 17.0 to 10.5 us (13.6 to 8.9 with a wide M=1 kernel), INT8
    from 26.5 to 16.7 us.
    """
    quantization = getattr(config, "quantization", None)
    return (
        _is_quantized(config)
        and not _float_attention_in_quantized_checkpoint(config)
        and quantization.weight_format is QuantizedWeightFormat.INTEGER_AFFINE
        and quantization.bits <= 8
        and not quantization.float_zero_point
        and not quantization.overrides
    )


def _use_fused_qkv(config: ArchitectureConfig) -> bool:
    """Whether the decoder layers take one packed Q/K/V projection.

    Float attention joins the checkpoint's matrices (fused-MoE or expert-only
    quantized checkpoints); quantized attention joins its integer blocks after
    conversion (:func:`_packs_quantized_qkv`). Only the generic
    :class:`~mobius.models.moe.MoEDecoderLayer` fallback keeps three projections.
    """
    return _use_fused_moe(config) or (
        _is_quantized(config)
        and (_float_attention_in_quantized_checkpoint(config) or _packs_quantized_qkv(config))
    )


_ATTENTION_PROJECTIONS = (".self_attn.qkv_proj", ".self_attn.o_proj")


def _attention_linear_class(config: ArchitectureConfig) -> type:
    """Projection class for the thinker's attention.

    ``export_attn_weight_bits`` turns the two projections into
    ``com.microsoft::MatMulNBits``, quantized from the checkpoint at export
    time. Otherwise they stay float, and
    :class:`~mobius.components.TransposedLinear` hands cuBLAS the layout it is
    fast at -- a choice the quantized path does not need, since
    ``MatMulNBits`` already takes the weight as ``[out, in]``.
    """
    bits = config.export_attn_weight_bits
    if bits is None:
        return TransposedLinear
    return make_quantized_linear_factory(
        bits=bits,
        block_size=config.export_weight_quant_block_size,
        has_zero_point=False,
    )


def _quantize_named_weight(
    state_dict: dict[str, torch.Tensor], prefix: str, *, bits: int, block_size: int
) -> dict[str, torch.Tensor]:
    """Replace one float weight with a ``MatMulNBits`` payload and its scales."""
    if bits != 8:
        raise ValueError(f"only 8-bit weight quantization is implemented, got {bits}")
    key = f"{prefix}.weight"
    if key not in state_dict:
        raise ValueError(f"{key} is not in the mapped weights, so it cannot be quantized")
    payload, scales = quantize_linear_int8_blocks(state_dict.pop(key), block_size=block_size)
    state_dict[key] = payload
    state_dict[f"{prefix}.scales"] = scales
    return state_dict


def _lm_head_class(config: ArchitectureConfig) -> type:
    """Projection class for the LM head.

    ``export_lm_head_weight_bits`` makes it a ``MatMulNBits`` too. It is one
    GEMV per step against the projections' 48, so a smaller prize -- 0.39 ms
    of a 4.86 ms decode step on Blackwell, and it already ran at 1.6 TB/s --
    but a cheaper one than the projections: the head applies once, where a
    projection's quantization error compounds through 48 layers.
    """
    bits = config.export_lm_head_weight_bits
    if bits is None:
        return Linear
    if config.tie_word_embeddings:
        raise ValueError(
            "export_lm_head_weight_bits cannot be used with tied word "
            "embeddings: quantizing the head would untie it from the "
            "embedding table and store the vocabulary twice"
        )
    return make_quantized_linear_factory(
        bits=bits,
        block_size=config.export_weight_quant_block_size,
        has_zero_point=False,
    )


def _quantize_attention_projections(
    state_dict: dict[str, torch.Tensor], *, bits: int, block_size: int
) -> dict[str, torch.Tensor]:
    """Replace the decoder's float projections with ``MatMulNBits`` payloads.

    Runs on the mapped names, after the Q/K/V tensors have been joined, so
    there are two weights per layer to convert.
    """
    if bits != 8:
        raise ValueError(f"export_attn_weight_bits must be 8 or None, got {bits}")
    converted: dict[str, torch.Tensor] = {}
    for key in list(state_dict):
        if not key.startswith("decoder.layers."):
            continue
        prefix = key.removesuffix(".weight")
        if prefix == key or not prefix.endswith(_ATTENTION_PROJECTIONS):
            continue
        payload, scales = quantize_linear_int8_blocks(
            state_dict.pop(key), block_size=block_size
        )
        converted[key] = payload
        converted[f"{prefix}.scales"] = scales
    if not converted:
        raise ValueError(
            "export_attn_weight_bits is set but the checkpoint has no float "
            "decoder.layers.*.self_attn.{qkv_proj,o_proj}.weight to quantize"
        )
    state_dict.update(converted)
    return state_dict


def _pack_qkv_weights(
    state_dict: dict[str, torch.Tensor], layers_prefix: str = "model.layers."
) -> dict[str, torch.Tensor]:
    """Join each decoder layer's separate Q/K/V projections into one tensor.

    :class:`~mobius.components.FusedQKVAttention` reads ``[Q | K | V]`` rows
    from a single ``qkv_proj``; the released checkpoints store the three apart.
    Joining them here turns three ``MatMul`` nodes per layer into one, which on
    the 48-layer thinker is 96 fewer nodes in every decode step.

    Every tensor of a projection is joined along dim 0: float weights before
    mapping (``model.layers.*``), and ``MatMulNBits`` weights / scales / zero
    points -- all rows of output features -- after quantized conversion
    (``layers_prefix="decoder.layers."``). Raw GPTQ / AWQ tensors put the
    output features last and must not be joined here. Only decoder layers are
    touched: the audio tower has its own attention and keeps three projections.
    """
    packed: dict[str, torch.Tensor] = {}
    for key in list(state_dict):
        if not key.startswith(layers_prefix):
            continue
        prefix, separator, suffix = key.partition(".self_attn.q_proj.")
        if not separator:
            continue
        parts = []
        for projection in ("q_proj", "k_proj", "v_proj"):
            part = state_dict.pop(f"{prefix}.self_attn.{projection}.{suffix}", None)
            if part is None:
                raise ValueError(
                    f"{prefix}.self_attn is missing {projection}.{suffix}; "
                    "the thinker's attention must ship all three projections"
                )
            parts.append(part)
        packed[f"{prefix}.self_attn.qkv_proj.{suffix}"] = torch.cat(parts, dim=0)
    state_dict.update(packed)
    return state_dict


def _interleave_gate_up(gate_up: torch.Tensor) -> torch.Tensor:
    """``[..., 2 * inter, hidden]`` gate-then-up rows → ``swiglu_fusion=1`` order.

    The fused kernel expects ``[g_0, u_0, g_1, u_1, ...]``. Interleaving here
    rather than in the graph keeps the 60 GB of expert weights out of ORT's
    constant folding at session load.
    """
    experts, fc1_out, hidden = gate_up.shape
    return (
        gate_up.reshape(experts, 2, fc1_out // 2, hidden)
        .transpose(1, 2)
        .reshape(experts, fc1_out, hidden)
    )


def _stack_expert_weights(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Rewrite HF expert weights into the fused MoE node's two parameters.

    Accepts both checkpoint layouts: the fused ``experts.gate_up_proj`` /
    ``experts.down_proj`` tensors that transformers produces, and the
    per-expert ``experts.{i}.{gate,up,down}_proj.weight`` tensors the released
    safetensors ship. Consumed entries are dropped as they are stacked so the
    per-expert copies do not stay resident.
    """
    per_expert: dict[str, dict[int, dict[str, torch.Tensor]]] = {}
    stacked: dict[str, torch.Tensor] = {}
    for key in list(state_dict):
        if ".mlp.experts." not in key:
            continue
        prefix, _, suffix = key.partition(".mlp.experts.")
        if suffix.startswith("gate_up_proj"):
            stacked[f"{prefix}.mlp.fc1_experts_weights"] = _interleave_gate_up(
                state_dict.pop(key)
            )
        elif suffix.startswith("down_proj"):
            stacked[f"{prefix}.mlp.fc2_experts_weights"] = state_dict.pop(key)
        else:
            index, _, projection = suffix.partition(".")
            if not index.isdigit():
                continue
            projection = projection.removesuffix(".weight")
            per_expert.setdefault(prefix, {}).setdefault(int(index), {})[projection] = (
                state_dict.pop(key)
            )

    for prefix, experts in per_expert.items():
        order = sorted(experts)
        gate_up = torch.stack(
            [
                torch.stack([experts[i]["gate_proj"], experts[i]["up_proj"]], dim=1).reshape(
                    -1, experts[i]["gate_proj"].shape[-1]
                )
                for i in order
            ]
        )
        stacked[f"{prefix}.mlp.fc1_experts_weights"] = gate_up
        stacked[f"{prefix}.mlp.fc2_experts_weights"] = torch.stack(
            [experts[i]["down_proj"] for i in order]
        )
        experts.clear()

    state_dict.update(stacked)
    return state_dict


class Qwen3OmniFusedMoE(nn.Module):
    """Routed experts of one decoder layer as a single fused MoE node.

    Expert weights are stored expert-major, matching the fused HF layout:
    ``fc1_experts_weights`` ``[E, 2 * moe_inter, hidden]`` with gate/up rows
    already interleaved for ``swiglu_fusion=1`` (see
    :meth:`Qwen3OmniThinkerForConditionalGeneration.preprocess_weights`), and
    ``fc2_experts_weights`` ``[E, hidden, moe_inter]``.

    The op's SwiGLU defaults (``activation_alpha=1.0``, ``activation_beta=0``,
    no clamp) are exactly Qwen3's ``silu(gate) * up``, so they are left unset.
    """

    def __init__(self, config: ArchitectureConfig, gate: nn.Module):
        super().__init__()
        assert config.num_local_experts is not None
        assert config.num_experts_per_tok is not None
        assert config.moe_intermediate_size is not None
        self.gate = gate
        self._hidden_size = config.hidden_size
        self._top_k = config.num_experts_per_tok
        self.fc1_experts_weights = nn.Parameter(
            [config.num_local_experts, 2 * config.moe_intermediate_size, config.hidden_size]
        )
        self.fc2_experts_weights = nn.Parameter(
            [config.num_local_experts, config.hidden_size, config.moe_intermediate_size]
        )

    def forward(self, op: OpBuilder, hidden_states: ir.Value):
        hidden_2d = op.Reshape(
            hidden_states, op.Constant(value_ints=[-1, self._hidden_size])
        )  # (batch * seq, hidden)
        # ``router_probs`` takes the raw logits: the kernel applies
        # softmax → top-k → optional renormalisation itself.
        router_logits, _routing_probs, normalize, routed_scaling = (
            _realize_gate_and_get_qmoe_routing(op, self.gate, hidden_2d)
        )
        # CastLike restores the input dtype: op.MoE is a contrib op whose
        # output carries no type, which would break downstream inference.
        expert_out = op.CastLike(
            op.MoE(  # type: ignore[attr-defined]
                hidden_2d,
                router_logits,
                self.fc1_experts_weights,
                None,  # fc1_experts_bias
                self.fc2_experts_weights,
                activation_type="swiglu",
                k=self._top_k,
                normalize_routing_weights=int(normalize),
                swiglu_fusion=1,
                _domain="com.microsoft",
            ),
            hidden_2d,
        )
        if routed_scaling != 1.0:  # noqa: RUF069
            expert_out = op.Mul(expert_out, op.CastLike(routed_scaling, expert_out))
        return op.Reshape(expert_out, op.Shape(hidden_states))


class Qwen3OmniDecoderLayer(MoEDecoderLayer):
    """Pre-norm decoder layer whose routed experts use the fused MoE node."""

    def __init__(self, config: ArchitectureConfig, gate: nn.Module, **kwargs):
        super().__init__(config, gate=gate, **kwargs)
        self.self_attn = FusedQKVAttention(
            config, linear_class=_attention_linear_class(config)
        )
        self.mlp = Qwen3OmniFusedMoE(config, gate=gate)


class Qwen3OmniQuantizedExpertsDecoderLayer(MoEDecoderLayer):
    """Decoder layer for expert-only quantized checkpoints.

    :class:`MoEDecoderLayer` quantizes attention whenever the config carries a
    quantization; here only the routed experts (QMoE) are quantized and the
    attention projections stay float.
    """

    def __init__(self, config: ArchitectureConfig, gate: nn.Module, **kwargs):
        super().__init__(config, gate=gate, **kwargs)
        self.self_attn = FusedQKVAttention(
            config, linear_class=_attention_linear_class(config)
        )


class Qwen3OmniQuantizedDecoderLayer(MoEDecoderLayer):
    """Decoder layer for checkpoints that quantize attention as well as the experts.

    The projections keep the checkpoint's integer blocks and the routed experts
    are one QMoE node, as :class:`MoEDecoderLayer` builds them, but Q/K/V are
    one ``MatMulNBits`` (:func:`_packs_quantized_qkv`) and the output projection
    another. ``fc1_rows_interleaved`` is for loaders that write QMoE's fc1 rows
    already gate/up-interleaved, so the graph does not reorder them.
    """

    def __init__(
        self,
        config: ArchitectureConfig,
        gate: nn.Module,
        *,
        fc1_rows_interleaved: bool = False,
        **kwargs,
    ):
        super().__init__(config, gate=gate, **kwargs)
        linear_class = _quantized_linear_class(config)
        self.self_attn = FusedQKVAttention(config, linear_class=linear_class)
        if fc1_rows_interleaved:
            self.mlp = MoELayer(
                config, gate=gate, linear_class=linear_class, fc1_rows_interleaved=True
            )


class Qwen3OmniPackQuantizedDecoderLayer(Qwen3OmniQuantizedDecoderLayer):
    """Decoder layer for compressed-tensors ``pack-quantized`` INT4 checkpoints.

    Its streaming plan joins Q/K/V and interleaves QMoE's fc1 rows as it writes them.
    """

    def __init__(self, config: ArchitectureConfig, gate: nn.Module, **kwargs):
        super().__init__(config, gate=gate, fc1_rows_interleaved=True, **kwargs)


class Qwen3OmniThinkerDecoderModel(Qwen3ASRDecoderModel):
    """Qwen3-Omni thinker text decoder: inputs_embeds → logits + KV cache.

    Same interface as :class:`Qwen3ASRDecoderModel`, with every layer's
    MLP replaced by a sparse MoE block. HF also supports dense layers via
    ``mlp_only_layers`` / ``decoder_sparse_step``; the released checkpoints
    use neither, so all layers are MoE here.
    """

    def __init__(self, config: ArchitectureConfig):
        # Skip Qwen3ASRDecoderModel.__init__, which builds dense layers.
        nn.Module.__init__(self)
        num_experts = config.num_local_experts
        top_k = config.num_experts_per_tok
        if num_experts is None or top_k is None:
            raise ValueError(
                "Qwen3-Omni decoder requires num_local_experts and num_experts_per_tok"
            )
        self._dtype = config.dtype
        if _use_fused_moe(config):
            layer_class = Qwen3OmniDecoderLayer
        elif _is_pack_quantized(config):
            if _float_attention_in_quantized_checkpoint(config):
                raise NotImplementedError(
                    "Qwen3-Omni compressed-tensors checkpoints are supported with "
                    "quantized attention only; this one leaves attention in float."
                )
            if config.export_attn_weight_bits is not None:
                raise ValueError(
                    "export_attn_weight_bits cannot be used with a compressed-tensors "
                    "checkpoint: its attention projections are already INT4."
                )
            layer_class = Qwen3OmniPackQuantizedDecoderLayer
        elif _is_quantized(config) and _float_attention_in_quantized_checkpoint(config):
            layer_class = Qwen3OmniQuantizedExpertsDecoderLayer
        elif _packs_quantized_qkv(config):
            layer_class = Qwen3OmniQuantizedDecoderLayer
        else:
            layer_class = MoEDecoderLayer
        self.layers = nn.ModuleList(
            [
                layer_class(
                    config,
                    gate=SoftmaxTopKGate(
                        config.hidden_size,
                        num_experts,
                        top_k,
                        norm_topk_prob=config.norm_topk_prob,
                    ),
                )
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = initialize_rope(config)
        self._gqa_fused_rope = getattr(config, "gqa_fused_rope", False)
        self.lm_head = _lm_head_class(config)(
            config.hidden_size, config.vocab_size, bias=False
        )


class Qwen3OmniThinkerForConditionalGeneration(nn.Module):
    """Qwen3-Omni thinker composite model (audio + text → text).

    Contains:
    - ``audio_tower``: Audio encoder (mel → audio features)
    - ``embedding``: Text+audio embedding fusion
    - ``decoder``: MoE text decoder with KV cache

    HuggingFace class: ``Qwen3OmniMoeThinkerForConditionalGeneration``
    """

    default_task: str = "speech-language"
    category: str = "Speech-to-Text"
    config_class: type = ArchitectureConfig

    # Runtime HF ``named_modules()`` sub-trees per ONNX component.
    HF_COMPONENT_SOURCES: ClassVar[dict[str, tuple[str, ...]]] = {
        "audio_encoder": ("thinker.audio_tower",),
        "embedding": ("thinker.model.embed_tokens",),
        "decoder": ("thinker.model.layers", "thinker.model.norm", "thinker.lm_head"),
    }

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self._fused_moe = _use_fused_moe(config)
        self.audio_tower = Qwen3ASRAudioEncoder(config)
        self.embedding = Qwen3ASREmbeddingModel(config)
        self.decoder = Qwen3OmniThinkerDecoderModel(config)

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ):
        """Text-only forward: embed ``input_ids`` and run the decoder."""
        inputs_embeds = self.embedding.embed_tokens(op, input_ids)
        return self.decoder(
            op,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )

    def build_pack_quantized_streaming_plan(
        self,
        component: str,
        key_index: Mapping[str, tuple[str, list[int], str]],
        initializers: Mapping[str, ir.Value],
    ) -> StreamingWeightPlan:
        """Bind a compressed-tensors ``pack-quantized`` INT4 checkpoint to one component.

        The attention projections' ``weight_packed`` words become ``MatMulNBits``
        blobs (Q/K/V joined into one) and each layer's per-expert projections
        become QMoE banks, by
        reinterpreting the packed bytes
        (:func:`~mobius.integrations.compressed_tensors.packed_weight_bytes`),
        never by requantizing; fc1 rows are written gate/up-interleaved, the
        order QMoE reads. Everything else is copied and cast to the export
        dtype, and ``--features lm-head-int8`` quantizes the float LM head as the
        float path does. Tensors are read only when the external data is
        written, so host memory holds one expert bank (~0.4 GB for the 30B
        thinker) rather than the 20 GB checkpoint.

        Every checkpoint key is classified: this component's tensors are
        targets; other components', the talker's, code2wav's and the vision
        tower's are ignored with the reason; ``weight_shape`` sidecars are
        ignored because the packed and scale shapes checked here determine
        them. A key that fits none of these is left unclassified, which the
        streaming loader rejects.
        """
        from onnx_ir import tensor_adapters

        from mobius.integrations._weight_loading import (
            StreamingTransformedBankSource,
            StreamingTransformedWeightSource,
            StreamingWeightPlan,
            StreamingWeightSource,
        )
        from mobius.integrations.compressed_tensors import packed_weight_bytes

        config = self.config
        if not _is_pack_quantized(config):
            raise ValueError(
                "build_pack_quantized_streaming_plan needs a pack-quantized config"
            )
        group = config.quantization.group_size
        num_experts = config.num_local_experts
        assert num_experts is not None
        prefix = "thinker." if any(key.startswith("thinker.") for key in key_index) else ""
        targets: dict[str, object] = {}
        bound: set[str] = set()

        def source(name: str) -> str:
            full = prefix + name
            if full not in key_index:
                raise ValueError(f"Checkpoint has no '{full}' for the {component} component")
            bound.add(full)
            return full

        def projection(
            modules: tuple[str, ...], part: str, init: ir.Value
        ) -> StreamingTransformedBankSource:
            """One MatMulNBits weight or scale tensor from one or more projections' rows.

            Q/K/V join along the output rows (``FusedQKVAttention``), which is
            exact for both the packed words and their per-row scales.
            """
            suffix = "weight_packed" if part == "weight" else "weight_scale"
            names = tuple(source(f"{m}.{suffix}") for m in modules)
            target_shape = tuple(int(d) for d in init.shape)
            cols = target_shape[1] * group  # n_blocks * group
            row_counts = [key_index[n][1][0] for n in names]
            if sum(row_counts) != target_shape[0]:
                raise ValueError(
                    f"{', '.join(names)} hold {sum(row_counts)} rows; the graph's "
                    f"{init.name} has {target_shape[0]}"
                )
            if part == "weight":
                shapes = tuple((rows, cols // 8) for rows in row_counts)
                dtype = "I32"

                def transform(tensors: list[torch.Tensor]) -> torch.Tensor:
                    return torch.cat([packed_weight_bytes(t) for t in tensors]).reshape(
                        target_shape
                    )
            else:
                shapes = tuple((rows, target_shape[1]) for rows in row_counts)
                dtype = key_index[names[0]][2]
                scale_dtype = tensor_adapters.to_torch_dtype(init.dtype)

                def transform(tensors: list[torch.Tensor]) -> torch.Tensor:
                    return torch.cat(tensors).to(scale_dtype)

            return StreamingTransformedBankSource(
                source_names=names,
                expected_source_shapes=shapes,
                expected_source_dtype=dtype,
                expected_target_shape=target_shape,
                expected_target_dtype=init.dtype,
                transform=transform,
            )

        def expert_bank(
            layer: str, kind: str, init: ir.Value
        ) -> StreamingTransformedBankSource:
            experts = f"model.layers.{layer}.mlp.experts"
            target_shape = tuple(int(d) for d in init.shape)
            fc1 = kind.startswith("fc1")
            suffix = "weight_packed" if kind.endswith("weights") else "weight_scale"
            projections = ("gate_proj", "up_proj") if fc1 else ("down_proj",)
            names = tuple(
                source(f"{experts}.{e}.{p}.{suffix}")
                for e in range(num_experts)
                for p in projections
            )
            rows = target_shape[1] // len(projections)
            if suffix == "weight_packed":
                source_shape = (rows, target_shape[2] // 4)
                source_dtype = "I32"
                convert = packed_weight_bytes
            else:
                source_shape = (rows, target_shape[2])
                source_dtype = key_index[names[0]][2]
                scale_dtype = tensor_adapters.to_torch_dtype(init.dtype)
                convert = lambda t: t.to(scale_dtype)  # noqa: E731

            def transform(
                tensors: list[torch.Tensor], n: int = len(projections)
            ) -> torch.Tensor:
                per_expert = [
                    # [rows, n, cols] -> [n * rows, cols]: rows g_0, u_0, g_1, u_1, ...
                    torch.stack(
                        [convert(t) for t in tensors[e * n : (e + 1) * n]], dim=1
                    ).reshape(-1, target_shape[2])
                    for e in range(num_experts)
                ]
                return torch.stack(per_expert)

            return StreamingTransformedBankSource(
                source_names=names,
                expected_source_shapes=(source_shape,) * len(names),
                expected_source_dtype=source_dtype,
                expected_target_shape=target_shape,
                expected_target_dtype=init.dtype,
                transform=transform,
            )

        def int8_head(part: str, init: ir.Value) -> StreamingTransformedWeightSource:
            name = source("lm_head.weight")
            block = config.export_weight_quant_block_size
            index = 0 if part == "weight" else 1
            out_dtype = tensor_adapters.to_torch_dtype(init.dtype)

            def transform(t: torch.Tensor, _n: str) -> torch.Tensor:
                quantized = quantize_linear_int8_blocks(t.to(torch.float32), block_size=block)
                return quantized[index].to(out_dtype)

            return StreamingTransformedWeightSource(
                source_name=name,
                expected_source_shape=tuple(key_index[name][1]),
                expected_source_dtype=key_index[name][2],
                expected_target_shape=tuple(int(d) for d in init.shape),
                expected_target_dtype=init.dtype,
                transform=transform,
            )

        for name, init in initializers.items():
            if init.const_value is not None:
                continue
            if name.startswith("audio_tower."):
                targets[name] = StreamingWeightSource(source(name))
            elif name == "embedding.embed_tokens.weight":
                targets[name] = StreamingWeightSource(source("model.embed_tokens.weight"))
            elif name == "decoder.norm.weight":
                targets[name] = StreamingWeightSource(source("model.norm.weight"))
            elif name in ("decoder.lm_head.weight", "decoder.lm_head.scales"):
                if config.export_lm_head_weight_bits is None:
                    targets[name] = StreamingWeightSource(source("lm_head.weight"))
                else:
                    targets[name] = int8_head(name.rsplit(".", 1)[1], init)
            elif match := re.fullmatch(r"decoder\.layers\.(\d+)\.(.+)", name):
                layer, rest = match.groups()
                module = f"model.layers.{layer}"
                if proj := re.fullmatch(
                    r"self_attn\.(qkv_proj|o_proj)\.(weight|scales)", rest
                ):
                    attn = f"{module}.self_attn"
                    modules = (
                        (f"{attn}.q_proj", f"{attn}.k_proj", f"{attn}.v_proj")
                        if proj.group(1) == "qkv_proj"
                        else (f"{attn}.o_proj",)
                    )
                    targets[name] = projection(modules, proj.group(2), init)
                elif rest.startswith("mlp.fc"):
                    targets[name] = expert_bank(layer, rest.removeprefix("mlp."), init)
                else:
                    targets[name] = StreamingWeightSource(source(f"{module}.{rest}"))
            else:
                raise ValueError(
                    f"No compressed-tensors source rule for '{name}' ({component})"
                )

        owners = (
            ("audio_tower.", "audio_encoder"),
            ("model.embed_tokens.", "embedding"),
            ("model.layers.", "decoder"),
            ("model.norm.", "decoder"),
            ("lm_head.", "decoder"),
        )
        computed = {
            name for name, init in initializers.items() if init.const_value is not None
        }
        ignored: dict[str, str] = {}
        for key in key_index:
            if key in bound:
                continue
            if key.startswith(_DROPPED_PREFIXES):
                ignored[key] = "outside the thinker (talker / code2wav)"
                continue
            inner = key.removeprefix(prefix)
            if inner.startswith(_DROPPED_THINKER_PREFIXES):
                ignored[key] = "vision tower is not exported"
            elif inner in computed:
                ignored[key] = "computed in the graph (e.g. the sinusoidal position table)"
            elif inner.endswith(".weight_shape"):
                ignored[key] = "logical shape; implied by the checked packed and scale shapes"
            else:
                owner = next((c for p, c in owners if inner.startswith(p)), None)
                if owner is not None and owner != component:
                    ignored[key] = f"bound by the {owner} component"
        return StreamingWeightPlan(
            targets=targets,
            ignored=ignored,
            report={
                "source_quantization": f"compressed-tensors pack-quantized int4 group {group}",
                "output_weight_format": (
                    "int4 MatMulNBits/QMoE, packed bytes reinterpreted"
                    if component == "decoder"
                    else "dense"
                ),
            },
        )

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Map HuggingFace weight names to ONNX module structure.

        - ``talker.*``, ``code2wav.*``, ``thinker.visual.*`` → dropped
          (the ``thinker.`` prefix is optional, so a bare thinker state dict
          also loads)
        - ``thinker.audio_tower.*`` → ``audio_tower.*``
        - ``thinker.model.embed_tokens.*`` → ``embedding.embed_tokens.*``
        - ``thinker.model.{layers,norm}.*`` → ``decoder.{layers,norm}.*``
          (expert weights are stacked for the fused MoE node, or split per
          expert for the loop fallback; Q/K/V are joined into ``qkv_proj``
          whenever the layers keep attention in float, and the projections are
          quantized here when ``export_attn_weight_bits`` asks for it)
        - ``thinker.lm_head.*`` → ``decoder.lm_head.*``
        """
        if _is_pack_quantized(self.config):
            raise NotImplementedError(
                "compressed-tensors pack-quantized checkpoints are bound by "
                "build_pack_quantized_streaming_plan (the transformers builder "
                "streams them); preprocess_weights does not load them."
            )
        thinker_state: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if key.startswith(_DROPPED_PREFIXES):
                continue
            key = key.removeprefix("thinker.")
            if key.startswith(_DROPPED_THINKER_PREFIXES):
                continue
            thinker_state[key] = value
        quantized = _is_quantized(self.config)
        # Quantized Q/K/V are joined after conversion, once every format is in
        # MatMulNBits' row-major [N, ...] layout (see the end of this method).
        quantized_qkv = _packs_quantized_qkv(self.config)
        if _use_fused_qkv(self.config) and not quantized_qkv:
            thinker_state = _pack_qkv_weights(thinker_state)
        if self._fused_moe:
            thinker_state = _stack_expert_weights(thinker_state)
        elif not quantized:
            thinker_state = _rename_moe_expert_weights(thinker_state)
        # Quantized checkpoints keep Olive's fused expert-major tensors
        # (``experts.gate_up_proj_qweight`` ...) for the QMoE packer below.

        cleaned: dict[str, torch.Tensor] = {}
        for key, value in thinker_state.items():
            if key.startswith("lm_head."):
                cleaned[f"decoder.{key}"] = value
                continue

            if key.startswith("model."):
                inner = key[len("model.") :]
                if inner.startswith("embed_tokens."):
                    cleaned[f"embedding.{inner}"] = value
                    continue
                if inner.startswith(("layers.", "norm.", "rotary_emb.")):
                    cleaned[f"decoder.{inner}"] = value
                    continue

            cleaned[key] = value

        embed_key = "embedding.embed_tokens.weight"
        lm_key = "decoder.lm_head.weight"
        if quantized:
            result = preprocess_quantized_weights(
                cleaned,
                self.config.quantization,
                tie_embeddings=self.config.tie_word_embeddings,
                embed_key=embed_key,
                head_key=lm_key,
                qmoe_target_path=".mlp",
                qmoe_quant_methods=("olive",),
                reject_quantized_embeddings_lm_head=True,
            )
        else:
            if self.config.tie_word_embeddings:
                if embed_key in cleaned and lm_key not in cleaned:
                    cleaned[lm_key] = cleaned[embed_key]
                elif lm_key in cleaned and embed_key not in cleaned:
                    cleaned[embed_key] = cleaned[lm_key]
            result = cleaned
        if quantized_qkv:
            result = _pack_qkv_weights(result, layers_prefix="decoder.layers.")

        # Last, so the checkpoint's own quantized tensors are already in their
        # final names and nothing here is mistaken for one of them.
        block_size = self.config.export_weight_quant_block_size
        if self.config.export_attn_weight_bits is not None:
            result = _quantize_attention_projections(
                result,
                bits=self.config.export_attn_weight_bits,
                block_size=block_size,
            )
        if self.config.export_lm_head_weight_bits is not None:
            result = _quantize_named_weight(
                result,
                "decoder.lm_head",
                bits=self.config.export_lm_head_weight_bits,
                block_size=block_size,
            )

        return result
