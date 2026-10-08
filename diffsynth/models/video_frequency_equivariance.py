import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

from .utils import load_state_dict


VIDEO_FREQUENCY_MODES = {
    "none",
    "easyvfx_faithful",
    "afldm_spatial_faithful",
    "video_temporal_extension",
}
VIDEO_FREQUENCY_MODE_CHOICES = tuple(sorted(VIDEO_FREQUENCY_MODES | {
    "easyvfx", "fei", "fera", "afldm", "fei_ea", "fera_ea", "equivariant",
}))
ATTENTION_IDLE = "idle"
ATTENTION_STORE = "store"
ATTENTION_LOAD = "load"


def _load_frequency_state_dict(checkpoint):
    checkpoint = str(checkpoint)
    marker = "video_frequency_adapter."
    if checkpoint.endswith(".safetensors"):
        state_dict = {}
        with safe_open(checkpoint, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if marker in key or key.startswith(("experts.", "router.", "mode_markers.")):
                    state_dict[key] = handle.get_tensor(key)
        return state_dict
    state_dict = load_state_dict(checkpoint)
    return {
        key: value
        for key, value in state_dict.items()
        if marker in key or key.startswith(("experts.", "router.", "mode_markers."))
    }


def _normalise_mode(mode):
    mode = str(mode or "none").strip().lower().replace("-", "_")
    aliases = {
        "easyvfx": "easyvfx_faithful",
        "fei": "easyvfx_faithful",
        "fera": "easyvfx_faithful",
        "afldm": "afldm_spatial_faithful",
        "fei_ea": "video_temporal_extension",
        "fera_ea": "video_temporal_extension",
        "equivariant": "video_temporal_extension",
    }
    mode = aliases.get(mode, mode)
    if mode not in VIDEO_FREQUENCY_MODES:
        raise ValueError(
            f"Unknown video_frequency_mode={mode}; expected one of {sorted(VIDEO_FREQUENCY_MODES)}"
        )
    return mode


def _parse_blocks(blocks, num_layers):
    if blocks is None or str(blocks).strip().lower() in {"", "all"}:
        return tuple(range(int(num_layers)))
    selected = []
    for item in str(blocks).replace("+", ",").split(","):
        item = item.strip()
        if not item:
            continue
        block_id = int(item)
        if block_id < 0 or block_id >= int(num_layers):
            raise ValueError(f"video-frequency block {block_id} is outside [0, {num_layers})")
        if block_id not in selected:
            selected.append(block_id)
    if not selected:
        raise ValueError("video_frequency_blocks did not select any transformer blocks")
    return tuple(selected)


class FrequencyEnergyIndicator(nn.Module):
    """EasyVFX-style appearance/motion frequency descriptor for BCTHW video latents."""

    def __init__(self, sigma_scales=(1.0, 2.0), eps=1e-8):
        super().__init__()
        if len(sigma_scales) != 2 or sigma_scales[0] >= sigma_scales[1]:
            raise ValueError("sigma_scales must contain two increasing values")
        self.sigma_scales = tuple(float(value) for value in sigma_scales)
        self.eps = float(eps)

    @staticmethod
    def _gaussian_kernel(sigma, channels, device, dtype):
        sigma = max(float(sigma), 1e-3)
        kernel_size = int(2 * 4 * sigma + 1) | 1
        coord = torch.arange(kernel_size, device=device, dtype=dtype)
        coord = coord - (kernel_size - 1) / 2
        kernel_1d = torch.exp(-0.5 * (coord / sigma).square())
        kernel_1d = kernel_1d / kernel_1d.sum().clamp_min(1e-12)
        kernel = torch.outer(kernel_1d, kernel_1d)
        return kernel.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)

    def _blur(self, tensor, sigma):
        channels = tensor.shape[1]
        weight = self._gaussian_kernel(sigma, channels, tensor.device, tensor.dtype)
        return F.conv2d(tensor, weight, padding=weight.shape[-1] // 2, groups=channels)

    def _band_energy(self, tensor):
        scale = min(tensor.shape[-2:]) / 128.0
        fine = self._blur(tensor, max(scale * self.sigma_scales[0], 1e-3))
        coarse = self._blur(tensor, max(scale * self.sigma_scales[1], 1e-3))
        bands = (coarse, fine - coarse, tensor - fine)
        energy = torch.stack([band.square().sum(dim=(1, 2, 3)) for band in bands], dim=-1)
        return energy / energy.sum(dim=-1, keepdim=True).clamp_min(self.eps)

    def forward(self, video):
        if video.ndim != 5:
            raise ValueError(f"FrequencyEnergyIndicator expects BCTHW, got {tuple(video.shape)}")
        video = video.float()
        appearance = video.mean(dim=2)
        if video.shape[2] > 1:
            frame_delta = video[:, :, 1:] - video[:, :, :-1]
            motion = torch.log1p(frame_delta.square().mean(dim=2))
        else:
            motion = torch.zeros_like(appearance)
        return torch.cat([self._band_energy(appearance), self._band_energy(motion)], dim=-1)


class SoftFrequencyRouter(nn.Module):
    def __init__(self, num_experts=3, hidden_dim=64, temperature=0.7):
        super().__init__()
        self.num_experts = int(num_experts)
        self.temperature = float(temperature)
        if self.num_experts < 2:
            raise ValueError("video frequency routing needs at least two experts")
        if self.temperature <= 0:
            raise ValueError("video frequency routing temperature must be positive")
        self.net = nn.Sequential(
            nn.Linear(6, int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(hidden_dim), self.num_experts),
        )

    def forward(self, descriptor):
        return torch.softmax(self.net(descriptor) / self.temperature, dim=-1)


class MultiExpertLoRA(nn.Module):
    """A parameter-budget preserving set of LoRA experts mixed per video sample."""

    def __init__(self, in_features, out_features, total_rank, num_experts, alpha=None):
        super().__init__()
        total_rank = int(total_rank)
        num_experts = int(num_experts)
        if total_rank < num_experts:
            raise ValueError("video_frequency_rank must be >= video_frequency_num_experts")
        base, remainder = divmod(total_rank, num_experts)
        self.ranks = tuple(base + (1 if index < remainder else 0) for index in range(num_experts))
        self.alpha = None if alpha is None else float(alpha)
        self.down = nn.ModuleList([nn.Linear(in_features, rank, bias=False) for rank in self.ranks])
        self.up = nn.ModuleList([nn.Linear(rank, out_features, bias=False) for rank in self.ranks])
        for rank, down, up in zip(self.ranks, self.down, self.up):
            nn.init.normal_(down.weight, std=1.0 / max(rank, 1))
            nn.init.zeros_(up.weight)

    def forward(self, hidden_states, routing_weights):
        if routing_weights is None:
            return torch.zeros_like(hidden_states)
        expert_outputs = []
        compute_dtype = self.down[0].weight.dtype
        hidden = hidden_states.to(compute_dtype)
        for rank, down, up in zip(self.ranks, self.down, self.up):
            delta = up(down(hidden))
            if self.alpha is not None:
                delta = delta * (self.alpha / max(rank, 1))
            expert_outputs.append(delta)
        stacked = torch.stack(expert_outputs, dim=1)
        weights = routing_weights[:, :, None, None].to(stacked.dtype)
        return (stacked * weights).sum(dim=1).to(hidden_states.dtype)


class AttentionExpertSet(nn.Module):
    def __init__(self, dim, rank, num_experts, alpha=None):
        super().__init__()
        self.q = MultiExpertLoRA(dim, dim, rank, num_experts, alpha=alpha)
        self.k = MultiExpertLoRA(dim, dim, rank, num_experts, alpha=alpha)
        self.v = MultiExpertLoRA(dim, dim, rank, num_experts, alpha=alpha)
        self.o = MultiExpertLoRA(dim, dim, rank, num_experts, alpha=alpha)


class VideoFrequencyEquivarianceAdapter(nn.Module):
    """Separated EasyVFX, spatial AF-LDM, and temporal video adaptations."""

    def __init__(
        self,
        dim,
        num_layers,
        mode="easyvfx_faithful",
        blocks="all",
        total_rank=4,
        num_experts=3,
        router_hidden_dim=64,
        router_temperature=0.7,
        equivariance_weight=0.0,
        equivariance_probability=1.0,
        temporal_shift=0.5,
        spatial_shift=0.5,
        temporal_cutoff=0.8,
        spatial_cutoff=0.9,
        cutoff_transition=0.1,
        spatial_shift_quantum=0.125,
        detach_reference=True,
        detach_equivariance_target=False,
    ):
        super().__init__()
        self.mode = _normalise_mode(mode)
        self.block_ids = _parse_blocks(blocks, num_layers)
        self.block_id_set = set(self.block_ids)
        marker_key = f"{self.mode}__blocks__{'_'.join(map(str, self.block_ids))}"
        # Trainable-only checkpoints need an explicit architecture identifier.
        # The zero marker participates in the graph but never changes the output.
        self.mode_markers = nn.ParameterDict({marker_key: nn.Parameter(torch.zeros(()))})
        if self.uses_frequency_experts:
            self.indicator = FrequencyEnergyIndicator()
            self.router = SoftFrequencyRouter(num_experts, router_hidden_dim, router_temperature)
            self.experts = nn.ModuleDict({
                str(block_id): AttentionExpertSet(dim, total_rank, num_experts, alpha=1.0)
                for block_id in self.block_ids
            })
        else:
            self.indicator = None
            self.router = None
            self.experts = nn.ModuleDict()
        self.equivariance_weight = float(equivariance_weight)
        self.equivariance_probability = float(equivariance_probability)
        self.temporal_shift = float(temporal_shift)
        self.spatial_shift = float(spatial_shift)
        self.temporal_cutoff = float(temporal_cutoff)
        self.spatial_cutoff = float(spatial_cutoff)
        self.cutoff_transition = float(cutoff_transition)
        self.spatial_shift_quantum = float(spatial_shift_quantum)
        self.detach_reference = bool(detach_reference)
        self.detach_equivariance_target = bool(detach_equivariance_target)
        self.last_descriptor = None
        self.last_routing_weights = None
        self.last_equivariance_loss = None
        self.last_shift = None
        self._reference_hidden = {}
        if not self.uses_equivariant_attention and self.equivariance_weight != 0:
            raise ValueError(
                "video frequency equivariance loss requires afldm_spatial_faithful "
                "or video_temporal_extension"
            )
        if not 0 <= self.equivariance_probability <= 1:
            raise ValueError("equivariance_probability must be in [0, 1]")

    @property
    def uses_frequency_experts(self):
        return self.mode in {"easyvfx_faithful", "video_temporal_extension"}

    @property
    def uses_equivariant_attention(self):
        return self.mode in {"afldm_spatial_faithful", "video_temporal_extension"}

    @property
    def uses_spatial_alias_filter(self):
        return self.mode in {"afldm_spatial_faithful", "video_temporal_extension"}

    @property
    def uses_temporal_extension(self):
        return self.mode == "video_temporal_extension"

    @property
    def uses_equivariance_loss(self):
        return self.uses_equivariant_attention and self.equivariance_weight > 0

    def route(self, video):
        if not self.uses_frequency_experts:
            return None
        descriptor = self.indicator(video).detach()
        weights = self.router(descriptor.to(dtype=next(self.router.parameters()).dtype))
        self.last_descriptor = descriptor.detach()
        self.last_routing_weights = weights.detach()
        return weights

    def projection_delta(self, block_id, projection, hidden_states, routing_weights):
        mode_marker = next(iter(self.mode_markers.values())).to(
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        if not self.uses_frequency_experts or int(block_id) not in self.block_id_set:
            return torch.zeros_like(hidden_states) + mode_marker * 0
        delta = getattr(self.experts[str(int(block_id))], projection)(hidden_states, routing_weights)
        return delta + mode_marker * 0

    def clear_attention_cache(self):
        self._reference_hidden.clear()

    @staticmethod
    def _match_batch(reference, current):
        if reference is None or reference.shape[0] == current.shape[0]:
            return reference
        if current.shape[0] % reference.shape[0] != 0:
            raise RuntimeError("Equivariant Attention reference/current batch sizes are incompatible")
        return reference.repeat_interleave(current.shape[0] // reference.shape[0], dim=0)

    def attention_context(self, block_id, hidden_states, routing_weights, attention_mode):
        """Return the hidden states used to form K/V, following AF-LDM EA."""
        block_id = int(block_id)
        if not self.uses_equivariant_attention or block_id not in self.block_id_set:
            return hidden_states, routing_weights
        if attention_mode == ATTENTION_STORE:
            reference = hidden_states.detach() if self.detach_reference else hidden_states
            routing = routing_weights
            if routing is not None and self.detach_reference:
                routing = routing.detach()
            self._reference_hidden[block_id] = (reference, routing)
            return hidden_states, routing_weights
        if attention_mode == ATTENTION_LOAD:
            if block_id not in self._reference_hidden:
                raise RuntimeError(f"missing Equivariant Attention reference hidden state for block {block_id}")
            reference, reference_routing = self._reference_hidden[block_id]
            reference = self._match_batch(reference, hidden_states)
            if reference_routing is not None:
                reference_routing = self._match_batch(reference_routing, routing_weights)
                reference_routing = reference_routing.to(
                    device=hidden_states.device,
                    dtype=routing_weights.dtype,
                )
            return reference.to(device=hidden_states.device, dtype=hidden_states.dtype), reference_routing
        return hidden_states, routing_weights

    @staticmethod
    def _raised_cosine(frequency, cutoff, transition):
        cutoff = min(max(float(cutoff), 0.0), 1.0)
        transition = min(max(float(transition), 0.0), 1.0)
        if transition == 0:
            return (frequency <= cutoff).to(frequency.dtype)
        end = min(cutoff + transition, 1.0)
        if end <= cutoff:
            return (frequency <= cutoff).to(frequency.dtype)
        phase = ((frequency - cutoff) / (end - cutoff)).clamp(0, 1)
        taper = 0.5 * (1.0 + torch.cos(math.pi * phase))
        return torch.where(
            frequency <= cutoff,
            torch.ones_like(taper),
            torch.where(frequency < end, taper, torch.zeros_like(taper)),
        )

    @staticmethod
    def _ideal_spatial_mask(height, width, cutoff_y, cutoff_x, device):
        fy = torch.fft.fftfreq(height, device=device, dtype=torch.float32).abs()
        fx = torch.fft.fftfreq(width, device=device, dtype=torch.float32).abs()
        keep_y = fy < (0.5 * float(cutoff_y))
        keep_x = fx < (0.5 * float(cutoff_x))
        return keep_y[:, None] & keep_x[None, :]

    def pre_patch_antialias(self, tensor, patch_size):
        """Ideal spatial low-pass before Wan's strided patch projection."""
        if not self.uses_spatial_alias_filter or tensor.ndim != 5:
            return tensor
        spatial_patch = tuple(int(value) for value in patch_size[-2:])
        if spatial_patch == (1, 1):
            return tensor
        cutoff_y = 1.0 / max(spatial_patch[0], 1)
        cutoff_x = 1.0 / max(spatial_patch[1], 1)
        mask = self._ideal_spatial_mask(
            tensor.shape[-2], tensor.shape[-1], cutoff_y, cutoff_x, tensor.device
        )
        spectrum = torch.fft.fft2(tensor.float(), dim=(-2, -1), norm="ortho")
        filtered = torch.fft.ifft2(spectrum * mask, dim=(-2, -1), norm="ortho").real
        return filtered.to(tensor.dtype)

    @staticmethod
    def spatial_fractional_shift(tensor, shift):
        """Circular, ideal Fourier translation over H/W only (AF-LDM Eq. 6)."""
        if tensor.ndim != 5:
            return tensor
        original_dtype = tensor.dtype
        _, dy, dx = [float(value) for value in shift]
        fy = torch.fft.fftfreq(tensor.shape[-2], device=tensor.device, dtype=torch.float32)
        fx = torch.fft.fftfreq(tensor.shape[-1], device=tensor.device, dtype=torch.float32)
        phase = torch.exp(-2j * math.pi * (fy[:, None] * dy + fx[None, :] * dx))
        spectrum = torch.fft.fft2(tensor.float(), dim=(-2, -1), norm="ortho")
        shifted = torch.fft.ifft2(spectrum * phase, dim=(-2, -1), norm="ortho").real
        return shifted.to(original_dtype)

    def bandlimited_shift(self, tensor, shift):
        if tensor.ndim != 5:
            return tensor
        if not self.uses_temporal_extension:
            return self.spatial_fractional_shift(tensor, shift)
        original_dtype = tensor.dtype
        _, _, frames, height, width = tensor.shape
        dt, dy, dx = [float(value) for value in shift]
        ft = torch.fft.fftfreq(frames, device=tensor.device, dtype=torch.float32)
        fy = torch.fft.fftfreq(height, device=tensor.device, dtype=torch.float32)
        fx = torch.fft.fftfreq(width, device=tensor.device, dtype=torch.float32)
        ft_norm = ft.abs() / max(float(ft.abs().max()), 1e-8)
        fy_grid, fx_grid = torch.meshgrid(fy, fx, indexing="ij")
        spatial_radius = torch.sqrt(fy_grid.square() + fx_grid.square())
        spatial_radius = spatial_radius / max(float(spatial_radius.max()), 1e-8)
        temporal_filter = self._raised_cosine(ft_norm, self.temporal_cutoff, self.cutoff_transition)
        spatial_filter = self._raised_cosine(spatial_radius, self.spatial_cutoff, self.cutoff_transition)
        low_pass = temporal_filter[:, None, None] * spatial_filter[None, :, :]
        phase = torch.exp(
            -2j * math.pi * (
                ft[:, None, None] * dt
                + fy[None, :, None] * dy
                + fx[None, None, :] * dx
            )
        )
        spectrum = torch.fft.fftn(tensor.float(), dim=(2, 3, 4), norm="ortho")
        shifted = torch.fft.ifftn(spectrum * low_pass * phase, dim=(2, 3, 4), norm="ortho").real
        return shifted.to(original_dtype)

    def sample_shift(self, device):
        def sample(maximum):
            if maximum <= 0:
                return 0.0
            return float(((torch.rand((), device=device) * 2 - 1) * maximum).item())

        def sample_quantised(maximum, quantum):
            if maximum <= 0:
                return 0.0
            steps = max(1, int(math.floor(maximum / max(quantum, 1e-8))))
            index = int(torch.randint(-steps, steps + 1, (), device=device).item())
            return float(index * quantum)

        if self.uses_temporal_extension:
            shift = (sample(self.temporal_shift), sample(self.spatial_shift), sample(self.spatial_shift))
        else:
            shift = (
                0.0,
                sample_quantised(self.spatial_shift, self.spatial_shift_quantum),
                sample_quantised(self.spatial_shift, self.spatial_shift_quantum),
            )
        self.last_shift = shift
        return shift

    @staticmethod
    def valid_mask(shape, shift, device):
        batch, _, frames, height, width = shape
        mask = torch.ones((batch, 1, frames, height, width), device=device, dtype=torch.float32)
        for dimension, offset in zip((2, 3, 4), shift):
            margin = int(math.ceil(abs(float(offset))))
            if margin <= 0:
                continue
            index = [slice(None)] * 5
            if offset >= 0:
                index[dimension] = slice(0, min(margin, shape[dimension]))
            else:
                index[dimension] = slice(max(0, shape[dimension] - margin), shape[dimension])
            mask[tuple(index)] = 0
        return mask

    def shift_model_inputs(self, inputs, shift):
        shifted = dict(inputs)
        reference = inputs.get("latents")
        if not torch.is_tensor(reference) or reference.ndim != 5:
            raise ValueError("video frequency equivariance requires BCTHW latent inputs")
        aligned_keys = {
            "latents", "y", "vace_context", "pose_latents", "motion_latents",
            "s2v_pose_latents", "control_camera_latents_input",
        }
        for key in aligned_keys:
            value = inputs.get(key)
            if torch.is_tensor(value) and value.ndim == 5 and value.shape[-3:] == reference.shape[-3:]:
                shifted[key] = self.bandlimited_shift(value, shift)
        return shifted

    def equivariance_loss(self, shifted_prediction, reference_prediction, shift):
        reference = (
            reference_prediction.detach()
            if self.detach_equivariance_target
            else reference_prediction
        )
        target = self.bandlimited_shift(reference, shift)
        mask = self.valid_mask(shifted_prediction.shape, shift, shifted_prediction.device)
        error = (shifted_prediction.float() - target.float()).square() * mask
        normaliser = mask.sum().clamp_min(1.0) * shifted_prediction.shape[1]
        loss = error.sum() / normaliser
        self.last_equivariance_loss = loss.detach()
        return loss


def attach_video_frequency_equivariance(
    pipe,
    mode="none",
    blocks="all",
    rank=4,
    num_experts=3,
    router_hidden_dim=64,
    router_temperature=0.7,
    equivariance_weight=0.0,
    equivariance_probability=1.0,
    temporal_shift=0.5,
    spatial_shift=0.5,
    temporal_cutoff=0.8,
    spatial_cutoff=0.9,
    cutoff_transition=0.1,
    spatial_shift_quantum=0.125,
    detach_reference=True,
    detach_equivariance_target=False,
    log_prefix="[video_frequency]",
):
    mode = _normalise_mode(mode)
    if mode == "none":
        return None
    dit = pipe.dit
    adapter = VideoFrequencyEquivarianceAdapter(
        dim=dit.dim,
        num_layers=len(dit.blocks),
        mode=mode,
        blocks=blocks,
        total_rank=rank,
        num_experts=num_experts,
        router_hidden_dim=router_hidden_dim,
        router_temperature=router_temperature,
        equivariance_weight=equivariance_weight,
        equivariance_probability=equivariance_probability,
        temporal_shift=temporal_shift,
        spatial_shift=spatial_shift,
        temporal_cutoff=temporal_cutoff,
        spatial_cutoff=spatial_cutoff,
        cutoff_transition=cutoff_transition,
        spatial_shift_quantum=spatial_shift_quantum,
        detach_reference=detach_reference,
        detach_equivariance_target=detach_equivariance_target,
    )
    adapter.to(device=pipe.device, dtype=pipe.torch_dtype)
    for parameter in adapter.parameters():
        parameter.requires_grad_(True)
    dit.video_frequency_adapter = adapter
    parameter_count = sum(parameter.numel() for parameter in adapter.parameters() if parameter.requires_grad)
    print(
        f"{log_prefix} attached mode={mode} blocks={adapter.block_ids} "
        f"frequency_experts={adapter.uses_frequency_experts} "
        f"equivariant_attention={adapter.uses_equivariant_attention} "
        f"temporal_extension={adapter.uses_temporal_extension} "
        f"rank={rank} experts={num_experts} trainable_params={parameter_count}",
        flush=True,
    )
    return adapter


def load_video_frequency_checkpoint(adapter, checkpoint, required=False, log_prefix="[video_frequency]"):
    if adapter is None or checkpoint is None:
        return False
    checkpoint = Path(checkpoint)
    state_dict = _load_frequency_state_dict(checkpoint)
    adapter_keys = set(adapter.state_dict().keys())
    prefixes = (
        "pipe.dit.video_frequency_adapter.",
        "dit.video_frequency_adapter.",
        "video_frequency_adapter.",
    )
    extracted = {}
    legacy_router_keys = {
        "router.net.1.weight": "router.net.0.weight",
        "router.net.1.bias": "router.net.0.bias",
        "router.net.3.weight": "router.net.2.weight",
        "router.net.3.bias": "router.net.2.bias",
    }
    for key, value in state_dict.items():
        candidate = key
        for prefix in prefixes:
            marker = key.find(prefix)
            if marker >= 0:
                candidate = key[marker + len(prefix):]
                break
        candidate = legacy_router_keys.get(candidate, candidate)
        if candidate in adapter_keys:
            extracted[candidate] = value
        if key in adapter_keys:
            extracted[key] = value
    if not extracted:
        if required:
            raise ValueError(f"{log_prefix} no adapter keys found in {checkpoint}")
        return False
    result = adapter.load_state_dict(extracted, strict=False)
    print(f"{log_prefix} checkpoint loaded: {checkpoint}, total {len(extracted)} keys", flush=True)
    if result.missing_keys:
        print(f"{log_prefix} warning: missing keys: {result.missing_keys}", flush=True)
    if result.unexpected_keys:
        print(f"{log_prefix} warning: unexpected keys: {result.unexpected_keys}", flush=True)
    return True


def infer_video_frequency_config(checkpoint):
    """Infer architecture mode and expert layout from a saved trainable checkpoint."""
    if checkpoint is None:
        return None
    state_dict = _load_frequency_state_dict(checkpoint)
    marker = "video_frequency_adapter."
    keys = []
    for key in state_dict:
        marker_index = key.find(marker)
        if marker_index >= 0:
            keys.append(key[marker_index + len(marker):])
        elif key.startswith(("experts.", "router.", "mode_markers.")):
            keys.append(key)
    if not keys:
        return None
    marker_key = next((key for key in keys if key.startswith("mode_markers.")), None)
    mode = "easyvfx_faithful"
    marker_blocks = []
    if marker_key is not None:
        encoded = marker_key.split(".", 1)[1]
        encoded_mode, separator, encoded_blocks = encoded.partition("__blocks__")
        if not separator:
            raise ValueError(f"invalid video frequency mode marker in {checkpoint}: {marker_key}")
        mode = _normalise_mode(encoded_mode)
        marker_blocks = [int(value) for value in encoded_blocks.split("_") if value]

    expert_blocks = sorted({
        int(parts[1])
        for key in keys
        if (parts := key.split("."))[:1] == ["experts"] and len(parts) > 2 and parts[1].isdigit()
    })
    block_ids = marker_blocks or expert_blocks
    if not block_ids:
        raise ValueError(f"incomplete video frequency adapter layout in {checkpoint}")
    config = {
        "mode": mode,
        "blocks": ",".join(str(block_id) for block_id in block_ids),
    }
    if mode == "afldm_spatial_faithful":
        return config

    def find_state_key(suffix):
        return next(
            (
                key
                for key in state_dict
                if key == suffix or key.endswith(marker + suffix)
            ),
            None,
        )

    router_hidden_key = find_state_key("router.net.0.weight") or find_state_key("router.net.1.weight")
    router_output_key = find_state_key("router.net.2.weight") or find_state_key("router.net.3.weight")
    if not expert_blocks or router_hidden_key is None or router_output_key is None:
        raise ValueError(f"incomplete video frequency expert layout in {checkpoint}")
    num_experts = int(state_dict[router_output_key].shape[0])
    rank = 0
    for expert_id in range(num_experts):
        suffix = f"experts.{expert_blocks[0]}.q.down.{expert_id}.weight"
        key = find_state_key(suffix)
        if key is None:
            raise ValueError(f"missing {suffix} in {checkpoint}")
        rank += int(state_dict[key].shape[0])
    config.update({
        "rank": rank,
        "num_experts": num_experts,
        "router_hidden_dim": int(state_dict[router_hidden_key].shape[0]),
    })
    return config
