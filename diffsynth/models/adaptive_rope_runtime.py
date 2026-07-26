import json
import torch
import weakref


SA_ROPE_MODES = {"none", "fixed", "source", "memory", "source+memory"}
SPATIAL_HEAD = 0
TEMPORAL_HEAD = 1


def _normalise_mode(mode):
    mode = str(mode or "none").strip().lower().replace("_", "+")
    if mode in {"source-memory", "source_memory", "source+mem", "src+memory"}:
        mode = "source+memory"
    if mode not in SA_ROPE_MODES:
        raise ValueError(f"Unknown sa_rope_mode={mode}; expected one of {sorted(SA_ROPE_MODES)}")
    return mode


def _tensor_stats(tensor, max_tokens=2048):
    if tensor is None or not torch.is_tensor(tensor):
        return None
    x = tensor.detach() if not tensor.requires_grad else tensor
    x = x.float()
    if x.numel() == 0:
        return None
    flat = x.reshape(x.shape[0], -1) if x.dim() > 1 else x.reshape(1, -1)
    if flat.shape[1] > max_tokens:
        idx = torch.linspace(0, flat.shape[1] - 1, max_tokens, device=flat.device).long()
        flat = flat[:, idx]
    mean_abs = flat.abs().mean(dim=1)
    std = flat.std(dim=1, unbiased=False)
    temporal_diff = torch.zeros_like(mean_abs)
    spatial_diff = torch.zeros_like(mean_abs)
    if x.dim() == 5:
        if x.shape[2] > 1:
            temporal_diff = (x[:, :, 1:] - x[:, :, :-1]).float().abs().mean(dim=(1, 2, 3, 4))
        dh = (x[:, :, :, 1:] - x[:, :, :, :-1]).float().abs().mean(dim=(1, 2, 3, 4)) if x.shape[3] > 1 else 0.0
        dw = (x[:, :, :, :, 1:] - x[:, :, :, :, :-1]).float().abs().mean(dim=(1, 2, 3, 4)) if x.shape[4] > 1 else 0.0
        spatial_diff = dh + dw if torch.is_tensor(dh) else torch.zeros_like(mean_abs)
    elif x.dim() >= 3 and x.shape[1] > 1:
        temporal_diff = (x[:, 1:] - x[:, :-1]).float().abs().mean(dim=tuple(range(1, x.dim())))
    return torch.stack([mean_abs, std, temporal_diff, spatial_diff], dim=-1)


def _memory_adapters(pipe):
    adapters = []
    text_adapter = getattr(pipe, "latent_memory_text", None)
    if text_adapter is not None:
        adapters.append(text_adapter)
    vace = getattr(pipe, "vace", None)
    if vace is not None:
        for name in ("latent_memory_context", "latent_memory_hint"):
            adapter = getattr(vace, name, None)
            if adapter is not None:
                adapters.append(adapter)
    return adapters


def _memory_stats(pipe, device, dtype):
    memory_values = []
    token_cosines = []
    route_probs = []
    for adapter in _memory_adapters(pipe):
        memory = getattr(adapter, "memory", None)
        if memory is not None:
            memory = memory.to(device=device, dtype=torch.float32)
            memory_values.append(memory.reshape(-1))
            memory_tokens = memory.reshape(-1, memory.shape[-1])
            if memory_tokens.shape[0] > 1:
                normed = torch.nn.functional.normalize(memory_tokens, dim=-1)
                cosine = normed @ normed.transpose(0, 1)
                mask = ~torch.eye(cosine.shape[0], device=cosine.device, dtype=torch.bool)
                if mask.any():
                    token_cosines.append(cosine[mask].mean())
        probs = getattr(adapter, "last_route_probs", None)
        if probs is not None:
            route_probs.append(probs.to(device=device, dtype=torch.float32).reshape(-1, probs.shape[-1]))
    if not memory_values:
        return torch.zeros(1, 4, device=device, dtype=dtype)
    memory_flat = torch.cat(memory_values, dim=0)
    mean_abs = memory_flat.abs().mean()
    std = memory_flat.std(unbiased=False)
    token_cos = torch.stack(token_cosines).mean() if token_cosines else memory_flat.new_tensor(0.0)
    if route_probs:
        usage = torch.cat(route_probs, dim=0).mean(dim=0)
        entropy = -(usage * usage.clamp_min(1e-6).log()).sum() / max(1, usage.numel())
    else:
        entropy = memory_flat.new_tensor(0.0)
    stats = torch.stack([mean_abs, std, token_cos, entropy]).view(1, 4)
    return stats.to(dtype=dtype)


class AdaptiveRoPEAdapter(torch.nn.Module):
    """Head-aware AST-RoPE controller.

    The learned modes are identity-initialized. The predicted coefficients are:
    temporal scale for temporal heads, spatial scale for h/w axes, and first-frame
    anchor scale for spatial heads' temporal axis.
    """

    def __init__(
        self,
        mode="none",
        hidden_dim=128,
        init_scale=0.1,
        fixed_temporal_scale=1.0,
        fixed_spatial_scale=1.0,
        min_scale=0.5,
        max_scale=1.5,
    ):
        super().__init__()
        self.mode = _normalise_mode(mode)
        self.init_scale = float(init_scale)
        self.fixed_temporal_scale = float(fixed_temporal_scale)
        self.fixed_spatial_scale = float(fixed_spatial_scale)
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)
        self.last_scales = None
        self._last_scales_for_loss = None
        self.last_features = None
        self.trace_head_roles = False
        self.trace_max_spatial_tokens = 64
        self.hidden_trace_max_tokens = 256
        self._attention_role_votes = {}
        self._hidden_traces = {}
        self._active_trace_tag = None
        if self.mode in {"source", "memory", "source+memory"}:
            hidden_dim = int(hidden_dim) if hidden_dim and hidden_dim > 0 else 128
            self.predictor = torch.nn.Sequential(
                torch.nn.LayerNorm(8),
                torch.nn.Linear(8, hidden_dim),
                torch.nn.SiLU(),
                torch.nn.Linear(hidden_dim, 3),
            )
            torch.nn.init.zeros_(self.predictor[-1].weight)
            torch.nn.init.zeros_(self.predictor[-1].bias)
        else:
            self.predictor = None
        self.register_buffer("head_roles", torch.empty(0, 0, dtype=torch.long), persistent=True)

    @property
    def enabled(self):
        return self.mode != "none"

    def _features(self, source_latents, pipe=None):
        if pipe is None and hasattr(self, "_pipe_ref"):
            pipe = self._pipe_ref()
        device = None
        dtype = torch.float32
        if torch.is_tensor(source_latents):
            device = source_latents.device
            dtype = source_latents.dtype if source_latents.dtype.is_floating_point else torch.float32
        elif self.predictor is not None:
            device = next(self.predictor.parameters()).device
            dtype = next(self.predictor.parameters()).dtype
        source = _tensor_stats(source_latents)
        if source is None:
            source = torch.zeros(1, 4, device=device, dtype=dtype)
        source = source.to(device=device, dtype=dtype)
        memory = _memory_stats(pipe, source.device, dtype) if pipe is not None else torch.zeros(1, 4, device=source.device, dtype=dtype)
        if memory.shape[0] != source.shape[0]:
            memory = memory.expand(source.shape[0], -1)
        if self.mode == "source":
            memory = torch.zeros_like(memory)
        elif self.mode == "memory":
            source = torch.zeros_like(source)
        return torch.cat([source, memory], dim=-1)

    def forward(self, source_latents=None, pipe=None):
        if pipe is None and hasattr(self, "_pipe_ref"):
            pipe = self._pipe_ref()
        if self.mode == "none":
            return None
        if self.mode == "fixed":
            device = source_latents.device if torch.is_tensor(source_latents) else next(self.parameters(), torch.empty((), device="cpu")).device
            scales = torch.tensor(
                [self.fixed_temporal_scale, self.fixed_spatial_scale, self.fixed_spatial_scale],
                device=device,
                dtype=torch.float32,
            ).view(1, 3)
        else:
            features = self._features(source_latents, pipe=pipe)
            self.last_features = features.detach()
            delta = self.predictor(features.to(dtype=next(self.predictor.parameters()).dtype))
            scales = 1.0 + self.init_scale * torch.tanh(delta.float())
        scales = scales.clamp(self.min_scale, self.max_scale)
        self.last_scales = scales.detach()
        self._last_scales_for_loss = scales
        return scales

    def regularization_loss(self):
        if self._last_scales_for_loss is None:
            return None
        return (self._last_scales_for_loss.float() - 1.0).pow(2).mean()

    def configure_head_roles(self, num_layers, num_heads, roles_path=None, default="alternate"):
        roles = None
        if roles_path:
            with open(roles_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "head_roles" in data:
                data = data["head_roles"]
            roles = torch.as_tensor(data, dtype=torch.long)
        if roles is None:
            roles = torch.zeros(int(num_layers), int(num_heads), dtype=torch.long)
            default = str(default or "alternate").lower()
            if default == "temporal":
                roles.fill_(TEMPORAL_HEAD)
            elif default == "spatial":
                roles.fill_(SPATIAL_HEAD)
            else:
                roles[:, 1::2] = TEMPORAL_HEAD
        if roles.shape != (int(num_layers), int(num_heads)):
            raise ValueError(f"head_roles shape {tuple(roles.shape)} does not match {(int(num_layers), int(num_heads))}")
        self.head_roles = roles.to(device=self.head_roles.device)

    def roles_for_block(self, block_id, num_heads, device):
        if self.head_roles.numel() == 0:
            roles = torch.zeros(int(num_heads), dtype=torch.long, device=device)
            roles[1::2] = TEMPORAL_HEAD
            return roles
        block_id = max(0, min(int(block_id or 0), self.head_roles.shape[0] - 1))
        return self.head_roles[block_id].to(device=device, dtype=torch.long)

    def begin_hidden_trace(self, tag):
        self._active_trace_tag = str(tag)
        self._hidden_traces[self._active_trace_tag] = {}

    def end_hidden_trace(self):
        self._active_trace_tag = None

    def pop_hidden_trace(self, tag):
        return self._hidden_traces.pop(str(tag), {})

    def record_hidden(self, block_id, hidden, grid_size=None):
        if self._active_trace_tag is None or hidden is None:
            return
        if grid_size is None:
            return
        f, h, w = [int(x) for x in grid_size]
        spatial = h * w
        token_count = min(hidden.shape[1], f * spatial)
        if token_count < f:
            return
        spatial = token_count // f
        tokens = hidden[:, :f * spatial].reshape(hidden.shape[0], f, spatial, hidden.shape[-1])
        sample_count = min(spatial, int(self.hidden_trace_max_tokens))
        if sample_count < spatial:
            idx = torch.linspace(0, spatial - 1, sample_count, device=hidden.device).long()
            tokens = tokens[:, :, idx]
        self._hidden_traces.setdefault(self._active_trace_tag, {})[int(block_id)] = {
            "tokens": tokens,
            "grid_size": (f, sample_count, 1),
        }

    def reset_attention_role_votes(self):
        self._attention_role_votes = {}

    def record_attention_density(self, block_id, q, k, grid_size):
        if not self.trace_head_roles or grid_size is None:
            return
        with torch.no_grad():
            f, h, w = [int(x) for x in grid_size]
            spatial_tokens = h * w
            if f <= 1 or spatial_tokens <= 0:
                return
            q = q.detach().float()
            k = k.detach().float()
            if q.dim() != 4 or k.dim() != 4:
                return
            usable = min(q.shape[1], f * spatial_tokens)
            q = q[:, :usable]
            k = k[:, :usable]
            spatial_tokens = usable // f
            if spatial_tokens <= 0:
                return
            q = q[:, :f * spatial_tokens].reshape(q.shape[0], f, spatial_tokens, q.shape[2], q.shape[3])
            k = k[:, :f * spatial_tokens].reshape(k.shape[0], f, spatial_tokens, k.shape[2], k.shape[3])
            sample_count = min(spatial_tokens, int(self.trace_max_spatial_tokens))
            idx = torch.linspace(0, spatial_tokens - 1, sample_count, device=q.device).long()
            q = q[:, :, idx]
            k = k[:, :, idx]
            scores = torch.einsum("bfshd,bgthd->bhfgst", q, k) / max(1.0, q.shape[-1] ** 0.5)
            attn = torch.softmax(scores.reshape(scores.shape[0], scores.shape[1], f, f, -1), dim=-1)
            density = (attn > (1.0 / max(1, sample_count * sample_count))).float().mean(dim=(0, 4))
            diag = density.diagonal(dim1=-2, dim2=-1).mean(dim=-1)
            non_diag_mask = ~torch.eye(f, device=density.device, dtype=torch.bool)
            off = density[:, non_diag_mask].reshape(density.shape[0], -1).max(dim=-1).values
            vote = (off > diag).long().cpu()
            self._attention_role_votes.setdefault(int(block_id), []).append(vote)

    def export_head_roles_from_votes(self):
        if not self._attention_role_votes:
            return None
        num_layers = max(self._attention_role_votes.keys()) + 1
        num_heads = len(next(iter(self._attention_role_votes.values()))[0])
        roles = torch.zeros(num_layers, num_heads, dtype=torch.long)
        for block_id, votes in self._attention_role_votes.items():
            stacked = torch.stack(votes, dim=0)
            roles[block_id] = (stacked.float().mean(dim=0) >= 0.5).long()
        return roles

    def save_head_roles_json(self, path):
        roles = self.export_head_roles_from_votes()
        if roles is None:
            roles = self.head_roles.detach().cpu()
        payload = {
            "legend": {"0": "spatial", "1": "temporal"},
            "head_roles": roles.tolist(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


def _scale_freqs(freqs, scale):
    if scale is None:
        return freqs
    scale = scale.to(device=freqs.device, dtype=torch.float64).mean()
    angles = torch.angle(freqs.to(torch.complex128)) * scale
    return torch.polar(torch.ones_like(angles), angles).to(freqs.dtype)


def _axis_freq_grid(model, f, h, w, scale_f=None, scale_h=None, scale_w=None):
    f_freqs = _scale_freqs(model.freqs[0][:f], scale_f)
    h_freqs = _scale_freqs(model.freqs[1][:h], scale_h)
    w_freqs = _scale_freqs(model.freqs[2][:w], scale_w)
    return torch.cat([
        f_freqs.view(f, 1, 1, -1).expand(f, h, w, -1),
        h_freqs.view(1, h, 1, -1).expand(f, h, w, -1),
        w_freqs.view(1, 1, w, -1).expand(f, h, w, -1),
    ], dim=-1).reshape(f * h * w, 1, -1)


def build_adaptive_rope_freqs(model, f, h, w, source_latents=None, pipe=None, block_id=None):
    adapter = getattr(model, "sa_rope_adapter", None)
    if adapter is None and pipe is not None:
        adapter = getattr(pipe, "sa_rope_adapter", None)
    scales = adapter(source_latents, pipe=pipe) if adapter is not None and adapter.enabled else None
    if scales is None:
        return _axis_freq_grid(model, f, h, w)
    temporal_scale = scales[:, 0]
    spatial_scale = scales[:, 1]
    anchor_scale = scales[:, 2]
    num_heads = int(getattr(model, "num_heads", 0) or len(adapter.roles_for_block(block_id or 0, 1, scales.device)))
    if num_heads <= 0:
        num_heads = int(getattr(getattr(model, "blocks", [None])[0].self_attn, "num_heads", 1)) if getattr(model, "blocks", None) else 1
    roles = adapter.roles_for_block(block_id or 0, num_heads, scales.device)
    spatial_freqs = _axis_freq_grid(model, f, h, w, scale_f=anchor_scale, scale_h=spatial_scale, scale_w=spatial_scale)
    temporal_freqs = _axis_freq_grid(model, f, h, w, scale_f=temporal_scale, scale_h=None, scale_w=None)
    spatial_freqs = spatial_freqs.expand(-1, num_heads, -1)
    temporal_freqs = temporal_freqs.expand(-1, num_heads, -1)
    role_mask = (roles.view(1, num_heads, 1) == TEMPORAL_HEAD).to(device=temporal_freqs.device)
    return torch.where(role_mask, temporal_freqs, spatial_freqs)


def attach_adaptive_rope(
    pipe,
    mode="none",
    hidden_dim=128,
    init_scale=0.1,
    fixed_temporal_scale=1.0,
    fixed_spatial_scale=1.0,
    min_scale=0.5,
    max_scale=1.5,
    head_roles_path=None,
    default_head_role="alternate",
    log_prefix="[sa_rope]",
):
    mode = _normalise_mode(mode)
    if mode == "none":
        return None
    dtype = getattr(pipe, "torch_dtype", torch.bfloat16)
    device = getattr(pipe, "device", "cpu")
    adapter = AdaptiveRoPEAdapter(
        mode=mode,
        hidden_dim=hidden_dim,
        init_scale=init_scale,
        fixed_temporal_scale=fixed_temporal_scale,
        fixed_spatial_scale=fixed_spatial_scale,
        min_scale=min_scale,
        max_scale=max_scale,
    )
    adapter.to(device=device, dtype=dtype)
    adapter._pipe_ref = weakref.ref(pipe)
    pipe.sa_rope_adapter = adapter
    if getattr(pipe, "dit", None) is not None:
        pipe.dit.sa_rope_adapter = adapter
    if getattr(pipe, "vace", None) is not None:
        pipe.vace.sa_rope_adapter = adapter
    if getattr(pipe, "dit", None) is not None and getattr(pipe.dit, "blocks", None) is not None:
        adapter.configure_head_roles(
            num_layers=len(pipe.dit.blocks),
            num_heads=getattr(pipe.dit, "num_heads", pipe.dit.blocks[0].self_attn.num_heads),
            roles_path=head_roles_path,
            default=default_head_role,
        )
    for param in adapter.parameters():
        param.requires_grad_(mode != "fixed")
    param_count = sum(param.numel() for param in adapter.parameters() if param.requires_grad)
    print(f"{log_prefix} attached mode={mode} trainable_params={param_count}", flush=True)
    return adapter


def load_adaptive_rope_checkpoint(adapter, checkpoint, required=False, log_prefix="[sa_rope]"):
    if adapter is None or checkpoint is None:
        return False
    from .utils import load_state_dict

    state_dict = load_state_dict(checkpoint)
    prefixes = ("pipe.sa_rope_adapter.", "sa_rope_adapter.", "dit.sa_rope_adapter.")
    adapter_state = adapter.state_dict()
    loaded = {}
    for key, value in state_dict.items():
        for prefix in prefixes:
            if key.startswith(prefix):
                stripped = key[len(prefix):]
                if stripped in adapter_state:
                    loaded[stripped] = value
        if key in adapter_state:
            loaded[key] = value
    if not loaded:
        if required:
            print(f"{log_prefix} warning: no sa_rope keys found in {checkpoint}", flush=True)
        return False
    result = adapter.load_state_dict(loaded, strict=False)
    print(f"{log_prefix} checkpoint loaded: {checkpoint}, total {len(loaded)} keys", flush=True)
    if result.missing_keys:
        print(f"{log_prefix} warning: missing keys: {result.missing_keys}", flush=True)
    if result.unexpected_keys:
        print(f"{log_prefix} warning: unexpected keys: {result.unexpected_keys}", flush=True)
    return True
