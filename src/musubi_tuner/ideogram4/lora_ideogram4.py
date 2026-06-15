"""
Custom LoRA network module for Ideogram 4.

Drop-in replacement for musubi_tuner.networks.lora that:
  1. Targets Ideogram4TransformerBlock instead of HunyuanVideo blocks, so all
     linear layers (attention.qkv, attention.o, feed_forward.w1/w2/w3) get LoRA.
  2. Saves weights in ComfyUI-native format:
       diffusion_model.layers.N.attention.qkv.lora_down.weight
     instead of musubi's default:
       lora_unet_layers_N_attention_qkv.lora_down.weight

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
# Ideogram 4 target modules
# ---------------------------------------------------------------------------

# LoRA is applied to all nn.Linear layers found inside each
# Ideogram4TransformerBlock, which gives:
#   layers.N.attention.qkv   (hidden → 3*hidden)
#   layers.N.attention.o     (hidden → hidden)
#   layers.N.feed_forward.w1 (hidden → intermediate)
#   layers.N.feed_forward.w2 (intermediate → hidden)
#   layers.N.feed_forward.w3 (hidden → intermediate)
IDEOGRAM4_TARGET_REPLACE_MODULES = ["Ideogram4TransformerBlock"]

# Internal prefix used during training (standard musubi convention).
# We rename keys to ComfyUI format at save time.
_MUSUBI_PREFIX = "lora_unet"

# ComfyUI expects this prefix in the LoRA file.
_COMFY_PREFIX = "diffusion_model"

# Exact module suffix names produced by musubi's naming for Ideogram 4.
_MODULE_MAP: Dict[str, str] = {
    "attention_qkv":   "attention.qkv",
    "attention_o":     "attention.o",
    "feed_forward_w1": "feed_forward.w1",
    "feed_forward_w2": "feed_forward.w2",
    "feed_forward_w3": "feed_forward.w3",
}

# Regex: lora_unet_layers_N_<module_suffix>
_MUSUBI_KEY_RE = re.compile(
    r"^lora_unet_layers_(\d+)_("
    + "|".join(re.escape(k) for k in _MODULE_MAP)
    + r")$"
)

# Reverse map for loading ComfyUI LoRAs back into training.
_COMFY_KEY_RE = re.compile(
    r"^diffusion_model\.layers\.(\d+)\."
    r"(attention\.qkv|attention\.o"
    r"|feed_forward\.w1|feed_forward\.w2|feed_forward\.w3)"
    r"$"
)
_COMFY_TO_MUSUBI: Dict[str, str] = {v: k for k, v in _MODULE_MAP.items()}


# ---------------------------------------------------------------------------
# Key conversion helpers
# ---------------------------------------------------------------------------

def _musubi_to_comfy(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Convert musubi lora_unet_* keys → ComfyUI diffusion_model.* keys."""
    out: Dict[str, torch.Tensor] = {}
    for key, value in sd.items():
        # key format: <lora_name>.<weight_type>
        # e.g. lora_unet_layers_0_attention_qkv.lora_down.weight
        dot = key.index(".") if "." in key else len(key)
        lora_name = key[:dot]
        suffix = key[dot:]  # e.g. ".lora_down.weight" or ".alpha" or ""

        m = _MUSUBI_KEY_RE.match(lora_name)
        if m:
            layer_idx, mod_suffix = m.group(1), m.group(2)
            module_path = _MODULE_MAP[mod_suffix]
            new_lora_name = f"{_COMFY_PREFIX}.layers.{layer_idx}.{module_path}"
            out[new_lora_name + suffix] = value
        else:
            # Fallback: keep as-is (shouldn't happen for Ideogram 4)
            out[key] = value

    return out


def _comfy_to_musubi(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Convert ComfyUI diffusion_model.* keys → musubi lora_unet_* keys."""
    out: Dict[str, torch.Tensor] = {}
    for key, value in sd.items():
        dot = key.index(".") if "." in key else len(key)
        # ComfyUI keys look like: diffusion_model.layers.0.attention.qkv.lora_down.weight
        # The lora_name part ends at the module boundary, before .lora_down / .lora_up / .alpha
        # Split on known suffixes
        for suf in (".lora_down.weight", ".lora_up.weight", ".alpha"):
            if key.endswith(suf):
                lora_path = key[: -len(suf)]  # e.g. diffusion_model.layers.0.attention.qkv
                suf_part = suf
                break
        else:
            out[key] = value
            continue

        m = _COMFY_KEY_RE.match(lora_path)
        if m:
            layer_idx, module_path = m.group(1), m.group(2)
            musubi_mod = _COMFY_TO_MUSUBI[module_path]
            new_lora_name = f"lora_unet_layers_{layer_idx}_{musubi_mod}"
            out[new_lora_name + suf_part] = value
        else:
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

        # Gather weights in musubi format, cast dtype.
        sd = self.state_dict()
        if dtype is not None:
            sd = {k: v.detach().clone().to("cpu").to(dtype) for k, v in sd.items()}
        else:
            sd = {k: v.detach().clone().to("cpu") for k, v in sd.items()}

        # Rename to ComfyUI format.
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
    """Create an Ideogram 4 LoRA network targeting the right transformer blocks."""

    # Remove the default HunyuanVideo exclude patterns for modulation layers;
    # Ideogram 4 doesn't have those.  Keep any user-supplied patterns.
    exclude_patterns = kwargs.pop("exclude_patterns", None) or []
    if isinstance(exclude_patterns, str):
        import ast
        exclude_patterns = ast.literal_eval(exclude_patterns)
    kwargs["exclude_patterns"] = exclude_patterns

    # Build via the standard factory, then swap the class.
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

    # Reattach as our subclass so save_weights is overridden.
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
    # Accept either ComfyUI (diffusion_model.*) or musubi (lora_unet_*) format.
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
