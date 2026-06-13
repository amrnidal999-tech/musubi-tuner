from pathlib import Path
from typing import List, Tuple

import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModel, AutoTokenizer

from musubi_tuner.ideogram4.pipeline import get_qwen3_vl_features, pad_text_features


DEFAULT_QWEN3_VL_PATH = "Qwen/Qwen3-VL-8B-Instruct"


def load_qwen3_vl_text_encoder(
    text_encoder_path: str = DEFAULT_QWEN3_VL_PATH,
    dtype: torch.dtype = torch.bfloat16,
    device_map="auto",
):
    """
    Load the frozen Qwen3-VL text encoder used by Ideogram4.

    Accepts either:
      - A HuggingFace model ID or local directory (standard from_pretrained path)
      - A single .safetensors file (e.g. Comfy-Org/Qwen3-VL qwen3vl_8b_bf16.safetensors)
        In this case the model architecture/config/tokenizer are loaded from HF
        (tiny JSON files only, no weight download) and the weights come from the
        local safetensors file.

    Ideogram4 does not use Qwen3-VL to generate text.
    It uses hidden states from Qwen3-VL as prompt-conditioning features.
    """
    path = Path(text_encoder_path)
    single_file = path.is_file() and path.suffix == ".safetensors"

    # Config/tokenizer source: HF for single-file mode, local/HF-ID otherwise.
    config_source = DEFAULT_QWEN3_VL_PATH if single_file else text_encoder_path

    print(f"Loading Qwen3-VL tokenizer from: {config_source}")
    tokenizer = AutoTokenizer.from_pretrained(config_source)

    if single_file:
        print(f"Single-file safetensors detected.")
        print(f"  Config source : {config_source} (HF — config JSON only, no weights)")
        print(f"  Weights source: {text_encoder_path}")

        config = AutoConfig.from_pretrained(config_source)

        # Build the model skeleton (random weights, no HF download of model weights).
        text_encoder = AutoModel.from_config(config, dtype=dtype)

        # Load the consolidated safetensors weights.
        print(f"  Loading state dict from safetensors...")
        state_dict = load_file(str(path), device="cpu")

        # Comfy-Org text encoder files store LM weights without the
        # 'language_model.' prefix that transformers' Qwen3-VL expects.
        # Remap: model.* / lm_head.* -> language_model.model.* / language_model.lm_head.*
        # Visual encoder keys are absent (not needed for text-only encoding).
        first_keys = list(state_dict.keys())[:5]
        needs_remap = any(k.startswith("model.") or k.startswith("lm_head.") for k in first_keys)
        if needs_remap:
            print(f"  Remapping Comfy-Org key format (adding 'language_model.' prefix)...")
            state_dict = {
                (f"language_model.{k}" if k.startswith("model.") or k.startswith("lm_head.") else k): v
                for k, v in state_dict.items()
            }

        missing, unexpected = text_encoder.load_state_dict(state_dict, strict=False)
        # After remapping, only visual encoder keys will be missing (not used for text encoding).
        lm_missing = [k for k in missing if not k.startswith("visual.")]
        if lm_missing:
            print(f"  Warning: {len(lm_missing)} non-visual missing keys — first few: {lm_missing[:3]}")
        else:
            print(f"  Language model weights loaded OK ({len(missing)} visual keys skipped — not needed).")

        text_encoder = text_encoder.to(dtype)

        # Device placement.
        if device_map == "auto":
            if torch.cuda.is_available():
                text_encoder = text_encoder.cuda()
        elif isinstance(device_map, (str, torch.device)):
            text_encoder = text_encoder.to(device_map)
    else:
        print(f"Loading Qwen3-VL text encoder from: {text_encoder_path}")
        text_encoder = AutoModel.from_pretrained(
            text_encoder_path,
            torch_dtype=dtype,
            device_map=device_map,
        )

    text_encoder.eval()
    text_encoder.requires_grad_(False)

    return tokenizer, text_encoder


def caption_to_token_ids(
    tokenizer,
    caption: str,
    max_text_length: int = 3072,
) -> List[int]:
    """
    Convert a caption/JSON prompt into Qwen3-VL token IDs using the Qwen chat template.

    Ideogram4 expects prompts to pass through Qwen3-VL in chat-template format.
    """
    messages = [{"role": "user", "content": [{"type": "text", "text": caption}]}]

    text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    ids = tokenizer(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_text_length,
    )["input_ids"]

    if len(ids) == 0:
        ids = [tokenizer.eos_token_id or 0]

    return ids


@torch.no_grad()
def encode_caption_to_features(
    tokenizer,
    text_encoder,
    caption: str,
    max_text_length: int = 3072,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Encode one caption into Ideogram4 Qwen3-VL features.

    Returns:
      (num_tokens, feature_dim)

    For Ideogram4, feature_dim is usually 53248 because multiple Qwen3-VL
    hidden layers are concatenated.
    """
    ids = caption_to_token_ids(tokenizer, caption, max_text_length=max_text_length)

    device = next(text_encoder.parameters()).device

    token_ids = torch.tensor([ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(token_ids)

    # Qwen3-VL expects 2D position ids for text-only mode here.
    pos_2d = (attention_mask.cumsum(dim=-1) - 1).clamp(min=0).to(torch.long)

    features = get_qwen3_vl_features(
        text_encoder,
        token_ids,
        attention_mask,
        pos_2d,
    )

    return features[0].to(dtype)


@torch.no_grad()
def encode_captions_to_padded_features(
    tokenizer,
    text_encoder,
    captions: List[str],
    max_text_length: int = 3072,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Encode a list of captions and pad them into a batch.

    Returns:
      llm_features: (B, max_tokens, feature_dim)
      text_mask:    (B, max_tokens)
    """
    features_list = [
        encode_caption_to_features(
            tokenizer,
            text_encoder,
            caption,
            max_text_length=max_text_length,
            dtype=dtype,
        )
        for caption in captions
    ]

    return pad_text_features(features_list, torch.device(device), dtype)
