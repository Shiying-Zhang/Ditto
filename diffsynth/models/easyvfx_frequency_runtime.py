import torch
import torch.nn.functional as F


EASYVFX_FREQUENCY_MODES = {"none", "fixed", "adaptive", "full"}


class EasyVFXFrequencyLoss(torch.nn.Module):
    """Frequency-driven auxiliary loss inspired by EasyVFX.

    The module is intentionally inference-free: it regularizes the denoising
    target during training and does not add parameters to the video editor.
    This keeps checkpoints compatible with the Wan+Ditto LoRA runtime.
    """

    def __init__(
        self,
        mode="none",
        low_ratio=0.25,
        temporal_low_ratio=0.35,
        low_weight=0.0,
        high_weight=0.0,
        temporal_weight=0.0,
        descriptor_weight=0.0,
        adaptive_weight=0.0,
        adaptive_temperature=0.7,
        detach_target_descriptor=True,
    ):
        super().__init__()
        mode = str(mode or "none")
        if mode not in EASYVFX_FREQUENCY_MODES:
            raise ValueError(f"Unknown easyvfx_frequency_mode={mode}; expected one of {sorted(EASYVFX_FREQUENCY_MODES)}")
        self.mode = mode
        self.low_ratio = float(low_ratio)
        self.temporal_low_ratio = float(temporal_low_ratio)
        self.low_weight = float(low_weight)
        self.high_weight = float(high_weight)
        self.temporal_weight = float(temporal_weight)
        self.descriptor_weight = float(descriptor_weight)
        self.adaptive_weight = float(adaptive_weight)
        self.adaptive_temperature = float(adaptive_temperature)
        self.detach_target_descriptor = bool(detach_target_descriptor)

    @property
    def enabled(self):
        return self.mode != "none" and any(
            weight != 0.0
            for weight in (
                self.low_weight,
                self.high_weight,
                self.temporal_weight,
                self.descriptor_weight,
                self.adaptive_weight,
            )
        )

    @staticmethod
    def _spatial_low_mask(tensor, ratio):
        height, width = int(tensor.shape[-2]), int(tensor.shape[-1])
        yy = torch.fft.fftfreq(height, device=tensor.device, dtype=torch.float32).view(height, 1)
        xx = torch.fft.fftfreq(width, device=tensor.device, dtype=torch.float32).view(1, width)
        radius = torch.sqrt(xx.pow(2) + yy.pow(2))
        cutoff = radius.max().clamp_min(1e-6) * float(ratio)
        mask = (radius <= cutoff).to(dtype=torch.float32)
        view_shape = [1] * tensor.dim()
        view_shape[-2:] = [height, width]
        return mask.view(*view_shape)

    def _spatial_split(self, tensor):
        tensor = tensor.float()
        fft = torch.fft.fftn(tensor, dim=(-2, -1), norm="ortho")
        mask = self._spatial_low_mask(tensor, self.low_ratio)
        low = torch.fft.ifftn(fft * mask, dim=(-2, -1), norm="ortho").real
        high = tensor - low
        return low, high

    def _temporal_high(self, tensor):
        tensor = tensor.float()
        if tensor.dim() < 5 or tensor.shape[2] < 3:
            return None
        frames = int(tensor.shape[2])
        fft = torch.fft.fft(tensor, dim=2, norm="ortho")
        freq = torch.fft.fftfreq(frames, device=tensor.device, dtype=torch.float32).abs()
        cutoff = freq.max().clamp_min(1e-6) * self.temporal_low_ratio
        mask = (freq > cutoff).to(dtype=torch.float32).view(1, 1, frames, 1, 1)
        return torch.fft.ifft(fft * mask, dim=2, norm="ortho").real

    @staticmethod
    def _energy(tensor):
        reduce_dims = tuple(range(1, tensor.dim()))
        return tensor.float().pow(2).mean(dim=reduce_dims)

    def _frequency_descriptor(self, tensor):
        low, high = self._spatial_split(tensor)
        temporal = self._temporal_high(tensor)
        if temporal is None:
            temporal_energy = torch.zeros_like(self._energy(low))
        else:
            temporal_energy = self._energy(temporal)
        descriptor = torch.stack(
            [
                self._energy(low),
                self._energy(high),
                temporal_energy,
            ],
            dim=-1,
        )
        return descriptor / descriptor.sum(dim=-1, keepdim=True).clamp_min(1e-6)

    def forward(self, pred_x0, target_x0):
        if not self.enabled:
            return None, {}
        pred = pred_x0.float()
        target = target_x0.to(device=pred.device, dtype=torch.float32)
        pred_low, pred_high = self._spatial_split(pred)
        target_low, target_high = self._spatial_split(target)

        low_loss = F.mse_loss(pred_low, target_low)
        high_loss = F.smooth_l1_loss(pred_high, target_high)
        pred_temporal = self._temporal_high(pred)
        target_temporal = self._temporal_high(target)
        if pred_temporal is None or target_temporal is None:
            temporal_loss = pred.new_zeros(())
        else:
            temporal_loss = F.smooth_l1_loss(pred_temporal, target_temporal)

        pred_descriptor = self._frequency_descriptor(pred)
        target_descriptor = self._frequency_descriptor(target)
        if self.detach_target_descriptor:
            target_descriptor = target_descriptor.detach()
        descriptor_loss = F.smooth_l1_loss(pred_descriptor, target_descriptor)

        fixed_loss = (
            self.low_weight * low_loss
            + self.high_weight * high_loss
            + self.temporal_weight * temporal_loss
            + self.descriptor_weight * descriptor_loss
        )
        adaptive_loss = pred.new_zeros(())
        if self.mode in {"adaptive", "full"} and self.adaptive_weight != 0.0:
            gates = torch.softmax(target_descriptor.detach() / max(self.adaptive_temperature, 1e-6), dim=-1)
            temporal_sample = (
                torch.zeros_like(self._sample_mse(pred_low, target_low))
                if pred_temporal is None or target_temporal is None
                else self._sample_smooth_l1(pred_temporal, target_temporal)
            )
            per_sample = torch.stack(
                [
                    self._sample_mse(pred_low, target_low),
                    self._sample_smooth_l1(pred_high, target_high),
                    temporal_sample,
                ],
                dim=-1,
            )
            adaptive_loss = (gates * per_sample).sum(dim=-1).mean() * self.adaptive_weight

        if self.mode == "adaptive":
            total = adaptive_loss + self.descriptor_weight * descriptor_loss
        else:
            total = fixed_loss + adaptive_loss
        metrics = {
            "low": low_loss,
            "high": high_loss,
            "temporal": temporal_loss,
            "descriptor": descriptor_loss,
            "adaptive": adaptive_loss,
        }
        return total, metrics

    @staticmethod
    def _sample_mse(pred, target):
        reduce_dims = tuple(range(1, pred.dim()))
        return (pred.float() - target.float()).pow(2).mean(dim=reduce_dims)

    @staticmethod
    def _sample_smooth_l1(pred, target):
        diff = F.smooth_l1_loss(pred.float(), target.float(), reduction="none")
        reduce_dims = tuple(range(1, diff.dim()))
        return diff.mean(dim=reduce_dims)


def build_easyvfx_frequency_loss(**kwargs):
    module = EasyVFXFrequencyLoss(**kwargs)
    return module if module.enabled else None
