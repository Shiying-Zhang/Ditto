import torch

from .latent_memory import (
    LatentMemoryContextAdapter,
    LatentMemoryHintAdapter,
    LatentMemoryTextAdapter,
    extract_latent_memory_state_dict,
)
from .utils import load_state_dict


def _parse_latent_memory_modes(mode):
    mode = str(mode or "none").strip()
    if not mode or mode == "none":
        return []
    if mode == "all":
        return ["text", "vace_context", "vace_hint"]
    modes = []
    for item in mode.replace(",", "+").split("+"):
        item = item.strip()
        if item and item != "none" and item not in modes:
            modes.append(item)
    return modes


def _attach_single_latent_memory(
    pipe,
    mode,
    num_tokens=4,
    hidden_dim=0,
    scale=1.0,
    init_std=0.02,
    log_prefix="[latent_memory]",
):
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


def attach_latent_memory(
    pipe,
    mode="none",
    num_tokens=4,
    hidden_dim=0,
    scale=1.0,
    init_std=0.02,
    log_prefix="[latent_memory]",
):
    modes = _parse_latent_memory_modes(mode)
    if not modes:
        return None, []
    adapters = []
    prefix_groups = []
    for item in modes:
        adapter, prefixes = _attach_single_latent_memory(
            pipe,
            item,
            num_tokens=num_tokens,
            hidden_dim=hidden_dim,
            scale=scale,
            init_std=init_std,
            log_prefix=log_prefix,
        )
        adapters.append(adapter)
        prefix_groups.append(prefixes)
    if len(adapters) == 1:
        return adapters[0], prefix_groups[0]
    bundle = torch.nn.ModuleList(adapters)
    bundle._latent_memory_prefix_groups = prefix_groups
    flat_prefixes = [prefix for group in prefix_groups for prefix in group]
    param_count = sum(param.numel() for param in bundle.parameters() if param.requires_grad)
    print(f"{log_prefix} attached combined mode={'+'.join(modes)} trainable_params={param_count}", flush=True)
    return bundle, flat_prefixes


def load_latent_memory_checkpoint(adapter, checkpoint, prefixes, required=False, log_prefix="[latent_memory]"):
    if adapter is None or checkpoint is None:
        return False
    state_dict = load_state_dict(checkpoint)
    prefix_groups = getattr(adapter, "_latent_memory_prefix_groups", None)
    if prefix_groups is not None:
        loaded_any = False
        for child, child_prefixes in zip(adapter, prefix_groups):
            loaded_any = load_latent_memory_checkpoint(
                child,
                checkpoint,
                child_prefixes,
                required=False,
                log_prefix=log_prefix,
            ) or loaded_any
        if required and not loaded_any:
            print(f"{log_prefix} warning: no latent_memory keys found in {checkpoint}", flush=True)
        return loaded_any
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
