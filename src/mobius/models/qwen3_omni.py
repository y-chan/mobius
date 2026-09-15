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

from mobius._configs import ArchitectureConfig
from mobius.components import (
    Linear,
    RMSNorm,
    SoftmaxTopKGate,
    initialize_rope,
)
from mobius.models.moe import MoEDecoderLayer, _rename_moe_expert_weights
from mobius.models.qwen3_asr import (
    Qwen3ASRAudioEncoder,
    Qwen3ASRDecoderModel,
    Qwen3ASREmbeddingModel,
)

# Checkpoint prefixes of components outside the thinker's audio + text path.
_DROPPED_PREFIXES = ("talker.", "code2wav.")
_DROPPED_THINKER_PREFIXES = ("visual.",)


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
        self.layers = nn.ModuleList(
            [
                MoEDecoderLayer(
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
          (fused ``experts.gate_up_proj`` / ``experts.down_proj`` are split
          per expert)
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
        thinker_state = _rename_moe_expert_weights(thinker_state)

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
        if self.config.tie_word_embeddings:
            if embed_key in cleaned and lm_key not in cleaned:
                cleaned[lm_key] = cleaned[embed_key]
            elif lm_key in cleaned and embed_key not in cleaned:
                cleaned[embed_key] = cleaned[lm_key]

        return cleaned
