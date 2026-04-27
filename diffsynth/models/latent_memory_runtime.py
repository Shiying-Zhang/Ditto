import torch

from .latent_memory import (
    LatentMemoryContextAdapter,
    LatentMemoryHintAdapter,
    LatentMemoryTextAdapter,
    extract_latent_memory_state_dict,
)
from .utils import load_state_dict


def attach_latent_memory(
    pipe,
    mode="none",
    num_tokens=4,
    hidden_dim=0,
    scale=1.0,
    init_std=0.02,
    log_prefix="[latent_memory]",
):
    mode = str(mode or "none")
    if mode == "none":
        return None, []
    dtype = getattr(pipe, "torch_dtype", torch.bfloat16)
    device = getattr(pipe, "device", "cpu")
    if mode == "text":
        text_dim = pipe.dit.text_embedding[0].in_features
        adapter = LatentMemoryTextAdapter(text_dim, num_tokens, hidden_dim, scale, init_std)
        pipe.latent_memory_text = adapter
        prefixes = ["pipe.latent_memory_text", "latent_memory_text"]
    elif mode == "vace_context":
        if getattr(pipe, "vace", None) is None:
            raise ValueError("latent_memory_mode=vace_context requires pipe.vace")
        adapter = LatentMemoryContextAdapter(pipe.dit.dim, num_tokens, hidden_dim, scale, init_std)
        pipe.vace.latent_memory_context = adapter
        prefixes = ["pipe.vace.latent_memory_context", "latent_memory_context"]
    elif mode == "vace_hint":
        if getattr(pipe, "vace", None) is None:
            raise ValueError("latent_memory_mode=vace_hint requires pipe.vace")
        adapter = LatentMemoryHintAdapter(pipe.dit.dim, len(pipe.vace.vace_layers), num_tokens, hidden_dim, scale, init_std)
        pipe.vace.latent_memory_hint = adapter
        prefixes = ["pipe.vace.latent_memory_hint", "latent_memory_hint"]
    else:
        raise ValueError(f"Unknown latent_memory_mode: {mode}")
    adapter.to(device=device, dtype=dtype)
    for param in adapter.parameters():
        param.requires_grad_(True)
    param_count = sum(param.numel() for param in adapter.parameters() if param.requires_grad)
    print(f"{log_prefix} attached mode={mode} trainable_params={param_count}", flush=True)
    return adapter, prefixes


def load_latent_memory_checkpoint(adapter, checkpoint, prefixes, required=False, log_prefix="[latent_memory]"):
    if adapter is None or checkpoint is None:
        return False
    state_dict = load_state_dict(checkpoint)
    adapter_state = adapter.state_dict()
    latent_state = extract_latent_memory_state_dict(state_dict, adapter_state.keys(), prefixes)
    if not latent_state:
        if required:
            print(f"{log_prefix} warning: no latent_memory keys found in {checkpoint}", flush=True)
        return False
    load_result = adapter.load_state_dict(latent_state, strict=False)
    print(f"{log_prefix} checkpoint loaded: {checkpoint}, total {len(latent_state)} keys", flush=True)
    if len(load_result[0]) > 0:
        print(f"{log_prefix} warning: missing latent_memory keys: {load_result[0]}", flush=True)
    if len(load_result[1]) > 0:
        print(f"{log_prefix} warning: unexpected latent_memory keys: {load_result[1]}", flush=True)
    return True
