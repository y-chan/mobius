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

Reference: https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct
HuggingFace class: Qwen3OmniMoeThinkerForConditionalGeneration
"""

from __future__ import annotations

from typing import ClassVar

import onnx_ir as ir
import torch
from onnxscript import OpBuilder, nn

from mobius._build_context import ep_capabilities
from mobius._configs import ArchitectureConfig
from mobius._weight_utils import preprocess_quantized_weights
from mobius.components import (
    Attention,
    Linear,
    RMSNorm,
    SoftmaxTopKGate,
    initialize_rope,
)
from mobius.components._moe import _realize_gate_and_get_qmoe_routing
from mobius.models.moe import MoEDecoderLayer, _rename_moe_expert_weights
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
        self.mlp = Qwen3OmniFusedMoE(config, gate=gate)


class Qwen3OmniQuantizedExpertsDecoderLayer(MoEDecoderLayer):
    """Decoder layer for expert-only quantized checkpoints.

    :class:`MoEDecoderLayer` quantizes attention whenever the config carries a
    quantization; here only the routed experts (QMoE) are quantized and the
    attention projections stay float.
    """

    def __init__(self, config: ArchitectureConfig, gate: nn.Module, **kwargs):
        super().__init__(config, gate=gate, **kwargs)
        self.self_attn = Attention(config)


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
        elif _is_quantized(config) and _float_attention_in_quantized_checkpoint(config):
            layer_class = Qwen3OmniQuantizedExpertsDecoderLayer
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
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)


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
          expert for the loop fallback)
        - ``thinker.lm_head.*`` → ``decoder.lm_head.*``
        """
        thinker_state: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if key.startswith(_DROPPED_PREFIXES):
                continue
            key = key.removeprefix("thinker.")
            if key.startswith(_DROPPED_THINKER_PREFIXES):
                continue
            thinker_state[key] = value
        quantized = _is_quantized(self.config)
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
            return preprocess_quantized_weights(
                cleaned,
                self.config.quantization,
                tie_embeddings=self.config.tie_word_embeddings,
                embed_key=embed_key,
                head_key=lm_key,
                qmoe_target_path=".mlp",
                qmoe_quant_methods=("olive",),
                reject_quantized_embeddings_lm_head=True,
            )
        if self.config.tie_word_embeddings:
            if embed_key in cleaned and lm_key not in cleaned:
                cleaned[lm_key] = cleaned[embed_key]
            elif lm_key in cleaned and embed_key not in cleaned:
                cleaned[embed_key] = cleaned[lm_key]

        return cleaned
