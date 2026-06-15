"""
Custom LoRA network module for Ideogram 4.

Drop-in replacement for musubi_tuner.networks.lora that:
  1. Targets Ideogram4Transformer2DModel (entire model) so all linear layers
     get LoRA — matching ai-toolkit's scope exactly.
  2. Saves weights in ComfyUI-native format:
       diffusion_model.layers.N.attention.qkv.lora_down.weight
     instead of musubi's default:
       lora_unet_layers_N_attention_qkv.lora_down.weight

Linear layers covered:
  Top-level:
    input_proj, llm_cond_proj,
    t_embedding.mlp_in, t_embedding.mlp_out,
    adaln_proj
  Per block (N = 0..33):
    layers.N.attention.qkv, layers.N.attention.o,
    layers.N.feed_forward.w1/w2/w3,
    layers.N.adaln_modulation
  Final layer:
    final_layer.linear, final_layer.adaln_modulation

Usage: pass --network_module musubi_tuner.ideogram4.lora_ideogram4 to the
training script instead of the default musubi_tuner.networks.lora.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

import torch
from safetensors.torch import save_file

from musubi_tuner.networks.lora import (
    LoRAModule,
    LoRANetwork,
    create_network,
    create_network_from_weights,
)

# ---------------------------------------------------------------------------
# Ideogram 4 target modules — full model so we match ai-toolkit scope
# ---------------------------------------------------------------------------

IDEOGRAM4_TARGET_REPLACE_MODULES = ["Ideogram4Transformer2DModel"]

# Internal prefix used during training (standard musubi convention).
_MUSUBI_PREFIX = "lora_unet"

# ComfyUI expects this prefix in the LoRA file.
_COMFY_PREFIX = "diffusion_model"

# ---------------------------------------------------------------------------
# Key maps: musubi suffix → ComfyUI module path
# ---------------------------------------------------------------------------

# Top-level and final-layer linears (not indexed by layer number).
# Key = musubi suffix after "lora_unet_"
# Value = ComfyUI path after "diffusion_model."
_STATIC_MODULE_MAP: Dict[str, str] = {
    "input_proj":                   "input_proj",
    "llm_cond_proj":                "llm_cond_proj",
    "t_embedding_mlp_in":           "t_embedding.mlp_in",
    "t_embedding_mlp_out":          "t_embedding.mlp_out",
    "adaln_proj":                   "adaln_proj",
    "final_layer_linear":           "final_layer.linear",
    "final_layer_adaln_modulation": "final_layer.adaln_modulation",
}

# Per-block linears (indexed by layer number N).
# Key = musubi suffix after "lora_unet_layers_N_"
# Value = ComfyUI path after "diffusion_model.layers.N."
_BLOCK_MODULE_MAP: Dict[str, str] = {
    "attention_qkv":    "attention.qkv",
    "attention_o":      "attention.o",
    "feed_forward_w1":  "feed_forward.w1",
    "feed_forward_w2":  "feed_forward.w2",
    "feed_forward_w3":  "feed_forward.w3",
    "adaln_modulation": "adaln_modulation",
}

# ---------------------------------------------------------------------------
# Compiled regexes
# ---------------------------------------------------------------------------

# Musubi → ComfyUI direction
_MUSUBI_BLOCK_RE = re.compile(
    r"^lora_unet_layers_(\d+)_("
    + "|".join(re.escape(k) for k in _BLOCK_MODULE_MAP)
    + r")$"
)
_MUSUBI_STATIC_RE = re.compile(
    r"^lora_unet_("
    + "|".join(re.escape(k) for k in _STATIC_MODULE_MAP)
    + r")$"
)

# ComfyUI → Musubi direction
_COMFY_BLOCK_RE = re.compile(
    r"^diffusion_model\.layers\.(\d+)\."
    + r"("
    + "|".join(re.escape(v) for v in _BLOCK_MODULE_MAP.values())
    + r")$"
)
_COMFY_STATIC_RE = re.compile(
    r"^diffusion_model\."
    + r"("
    + "|".join(re.escape(v) for v in _STATIC_MODULE_MAP.values())
    + r")$"
)

# Reverse lookups
_COMFY_BLOCK_TO_MUSUBI: Dict[str, str] = {v: k for k, v in _BLOCK_MODULE_MAP.items()}
_COMFY_STATIC_TO_MUSUBI: Dict[str, str] = {v: k for k, v in _STATIC_MODULE_MAP.items()}


# ---------------------------------------------------------------------------
# Key conversion helpers
# ---------------------------------------------------------------------------

def _split_lora_key(key: str):
    """Split 'lora_name.lora_down.weight' → ('lora_name', '.lora_down.weight')."""
    for suf in (".lora_down.weight", ".lora_up.weight", ".alpha"):
        if key.endswith(suf):
            return key[: -len(suf)], suf
    # Fallback: split at first dot
    dot = key.index(".") if "." in key else len(key)
    return key[:dot], key[dot:]


def _musubi_to_comfy(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Convert musubi lora_unet_* keys → ComfyUI diffusion_model.* keys."""
    out: Dict[str, torch.Tensor] = {}
    for key, value in sd.items():
        lora_name, suffix = _split_lora_key(key)

        # Per-block modules: lora_unet_layers_N_<suffix>
        m = _MUSUBI_BLOCK_RE.match(lora_name)
        if m:
            layer_idx, mod_suffix = m.group(1), m.group(2)
            module_path = _BLOCK_MODULE_MAP[mod_suffix]
            new_name = f"{_COMFY_PREFIX}.layers.{layer_idx}.{module_path}"
            out[new_name + suffix] = value
            continue

        # Top-level / final-layer modules: lora_unet_<suffix>
        m = _MUSUBI_STATIC_RE.match(lora_name)
        if m:
            mod_suffix = m.group(1)
            module_path = _STATIC_MODULE_MAP[mod_suffix]
            new_name = f"{_COMFY_PREFIX}.{module_path}"
            out[new_name + suffix] = value
            continue

        # Unknown key — keep as-is (should not happen for Ideogram 4)
        out[key] = value

    return out


def _comfy_to_musubi(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Convert ComfyUI diffusion_model.* keys → musubi lora_unet_* keys."""
    out: Dict[str, torch.Tensor] = {}
    for key, value in sd.items():
        lora_path, suf_part = _split_lora_key(key)
        if not suf_part:
            out[key] = value
            continue

        # Per-block: diffusion_model.layers.N.<module_path>
        m = _COMFY_BLOCK_RE.match(lora_path)
        if m:
            layer_idx, module_path = m.group(1), m.group(2)
            musubi_mod = _COMFY_BLOCK_TO_MUSUBI[module_path]
            new_name = f"lora_unet_layers_{layer_idx}_{musubi_mod}"
            out[new_name + suf_part] = value
            continue

        # Top-level / final: diffusion_model.<module_path>
        m = _COMFY_STATIC_RE.match(lora_path)
        if m:
            module_path = m.group(1)
            musubi_mod = _COMFY_STATIC_TO_MUSUBI[module_path]
            new_name = f"lora_unet_{musubi_mod}"
            out[new_name + suf_part] = value
            continue

        out[key] = value

    return out


# ---------------------------------------------------------------------------
# Custom LoRANetwork subclass that saves in ComfyUI format
# ---------------------------------------------------------------------------

class LoRANetworkIdeogram4(LoRANetwork):
    """LoRANetwork variant that saves/loads with ComfyUI-native key names."""

    def save_weights(self, file: str, dtype: Optional[torch.dtype], metadata: Optional[dict]):
        if metadata is not None and len(metadata) == 0:
            metadata = None

        sd = self.state_dict()
        if dtype is not None:
            sd = {k: v.detach().clone().to("cpu").to(dtype) for k, v in sd.items()}
        else:
            sd = {k: v.detach().clone().to("cpu") for k, v in sd.items()}

        sd_comfy = _musubi_to_comfy(sd)

        import os
        if os.path.splitext(file)[1] == ".safetensors":
            from musubi_tuner.utils import model_utils

            if metadata is None:
                metadata = {}
            model_hash, legacy_hash = model_utils.precalculate_safetensors_hashes(sd_comfy, metadata)
            metadata["sshs_model_hash"] = model_hash
            metadata["sshs_legacy_hash"] = legacy_hash

            save_file(sd_comfy, file, metadata)
        else:
            torch.save(sd_comfy, file)


# ---------------------------------------------------------------------------
# Public API — mirrors musubi_tuner.networks.lora
# ---------------------------------------------------------------------------

def create_arch_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae,
    text_encoders,
    unet,
    neuron_dropout: Optional[float] = None,
    **kwargs,
) -> LoRANetworkIdeogram4:
    """Create an Ideogram 4 LoRA network targeting the full transformer."""

    exclude_patterns = kwargs.pop("exclude_patterns", None) or []
    if isinstance(exclude_patterns, str):
        import ast
        exclude_patterns = ast.literal_eval(exclude_patterns)
    kwargs["exclude_patterns"] = exclude_patterns

    base_net = create_network(
        IDEOGRAM4_TARGET_REPLACE_MODULES,
        _MUSUBI_PREFIX,
        multiplier,
        network_dim,
        network_alpha,
        vae,
        text_encoders,
        unet,
        neuron_dropout=neuron_dropout,
        **kwargs,
    )

    base_net.__class__ = LoRANetworkIdeogram4
    return base_net


def create_arch_network_from_weights(
    multiplier: float,
    weights_sd: Dict[str, torch.Tensor],
    text_encoders=None,
    unet=None,
    for_inference: bool = False,
    **kwargs,
) -> LoRANetworkIdeogram4:
    """
    Load an Ideogram 4 LoRA from a ComfyUI-format safetensors file.
    Keys are converted from diffusion_model.* → lora_unet_* before passing
    to musubi's weight-based network builder.
    """
    first_key = next(iter(weights_sd))
    if first_key.startswith("diffusion_model."):
        weights_sd = _comfy_to_musubi(weights_sd)

    net = create_network_from_weights(
        IDEOGRAM4_TARGET_REPLACE_MODULES,
        multiplier,
        weights_sd,
        text_encoders,
        unet,
        for_inference=for_inference,
        **kwargs,
    )
    net.__class__ = LoRANetworkIdeogram4
    return net


def prepare_network(args):
    """Called by musubi trainer after network creation (no-op for us)."""
    pass
