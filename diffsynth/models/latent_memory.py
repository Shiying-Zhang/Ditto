import torch


def _default_hidden_dim(dim, hidden_dim):
    if hidden_dim and hidden_dim > 0:
        return hidden_dim
    return max(64, min(1024, dim // 4))


def _as_batched_sequence(tensor):
    if tensor.dim() == 2:
        return tensor.unsqueeze(0), True
    return tensor, False


def _mean_pool_sequence(context):
    context, squeezed = _as_batched_sequence(context)
    pooled = context.mean(dim=1)
    return pooled, squeezed


def _match_batch(tensor, batch_size):
    if tensor.shape[0] == batch_size:
        return tensor
    if tensor.shape[0] == 1:
        return tensor.expand(batch_size, -1)
    return tensor.mean(dim=0, keepdim=True).expand(batch_size, -1)


class LatentMemoryTextAdapter(torch.nn.Module):
    """Append prompt-conditioned latent memory tokens before Wan text projection."""

    def __init__(self, text_dim, num_tokens=4, hidden_dim=0, scale=1.0, init_std=0.02):
        super().__init__()
        if num_tokens <= 0:
            raise ValueError("num_tokens must be positive")
        hidden_dim = _default_hidden_dim(text_dim, hidden_dim)
        self.text_dim = int(text_dim)
        self.num_tokens = int(num_tokens)
        self.scale = float(scale)
        self.memory = torch.nn.Parameter(torch.empty(num_tokens, text_dim))
        self.gate = torch.nn.Sequential(
            torch.nn.LayerNorm(text_dim),
            torch.nn.Linear(text_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, num_tokens),
            torch.nn.Sigmoid(),
        )
        torch.nn.init.normal_(self.memory, mean=0.0, std=init_std)

    def forward(self, context):
        context, squeezed = _as_batched_sequence(context)
        pooled = context.mean(dim=1)
        gate = self.gate(pooled).to(dtype=context.dtype).unsqueeze(-1)
        memory = self.memory.to(dtype=context.dtype, device=context.device)
        memory = memory.unsqueeze(0).expand(context.shape[0], -1, -1)
        memory = memory * gate * self.scale
        context = torch.cat([context, memory], dim=1)
        return context.squeeze(0) if squeezed else context


class LatentMemoryContextAdapter(torch.nn.Module):
    """Inject prompt-conditioned latent memory into the VACE context branch."""

    def __init__(self, dim, num_tokens=4, hidden_dim=0, scale=1.0, init_std=0.02):
        super().__init__()
        if num_tokens <= 0:
            raise ValueError("num_tokens must be positive")
        hidden_dim = _default_hidden_dim(dim, hidden_dim)
        self.dim = int(dim)
        self.num_tokens = int(num_tokens)
        self.scale = float(scale)
        self.memory = torch.nn.Parameter(torch.empty(num_tokens, dim))
        self.selector = torch.nn.Sequential(
            torch.nn.LayerNorm(dim),
            torch.nn.Linear(dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, num_tokens),
        )
        torch.nn.init.normal_(self.memory, mean=0.0, std=init_std)

    def _memory_delta(self, context, batch_size):
        pooled, _ = _mean_pool_sequence(context)
        weights = torch.softmax(self.selector(pooled), dim=-1).to(dtype=context.dtype)
        memory = self.memory.to(dtype=context.dtype, device=context.device)
        delta = weights @ memory
        return _match_batch(delta, batch_size)

    def forward(self, vace_tokens, context):
        delta = self._memory_delta(context, vace_tokens.shape[0])
        delta = delta.to(dtype=vace_tokens.dtype, device=vace_tokens.device)
        return vace_tokens + delta.unsqueeze(1) * self.scale


class LatentMemoryHintAdapter(torch.nn.Module):
    """Inject layer-specific latent memory into VACE hints before main-branch fusion."""

    def __init__(self, dim, num_layers, num_tokens=4, hidden_dim=0, scale=1.0, init_std=0.02):
        super().__init__()
        if num_tokens <= 0:
            raise ValueError("num_tokens must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        hidden_dim = _default_hidden_dim(dim, hidden_dim)
        self.dim = int(dim)
        self.num_layers = int(num_layers)
        self.num_tokens = int(num_tokens)
        self.scale = float(scale)
        self.memory = torch.nn.Parameter(torch.empty(num_layers, num_tokens, dim))
        self.selector = torch.nn.Sequential(
            torch.nn.LayerNorm(dim),
            torch.nn.Linear(dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, num_tokens),
        )
        torch.nn.init.normal_(self.memory, mean=0.0, std=init_std)

    def forward(self, hint, context, hint_index=0, block_id=None):
        del block_id
        hint_index = int(hint_index)
        if hint_index < 0 or hint_index >= self.num_layers:
            raise IndexError(f"hint_index {hint_index} out of range for {self.num_layers} latent-memory layers")
        pooled, _ = _mean_pool_sequence(context)
        weights = torch.softmax(self.selector(pooled), dim=-1).to(dtype=hint.dtype)
        memory = self.memory[hint_index].to(dtype=hint.dtype, device=hint.device)
        delta = weights @ memory
        delta = _match_batch(delta, hint.shape[0])
        return hint + delta.unsqueeze(1) * self.scale


def extract_latent_memory_state_dict(state_dict, adapter_state_keys, prefixes):
    extracted = {}
    adapter_state_keys = set(adapter_state_keys)
    for key, value in state_dict.items():
        if key in adapter_state_keys:
            extracted[key] = value
            continue
        for prefix in prefixes:
            prefix = prefix.rstrip(".")
            prefix_with_dot = prefix + "."
            if key.startswith(prefix_with_dot):
                stripped = key[len(prefix_with_dot):]
                if stripped in adapter_state_keys:
                    extracted[stripped] = value
                break
    return extracted
