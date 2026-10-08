import unittest
from tempfile import TemporaryDirectory

import torch
from safetensors.torch import save_file

from diffsynth.models.video_frequency_equivariance import (
    ATTENTION_LOAD,
    ATTENTION_STORE,
    FrequencyEnergyIndicator,
    MultiExpertLoRA,
    SoftFrequencyRouter,
    VideoFrequencyEquivarianceAdapter,
    infer_video_frequency_config,
    _normalise_mode,
)
from diffsynth.models.wan_video_dit import add_projection_delta


class VideoFrequencyEquivarianceTest(unittest.TestCase):
    def test_projection_delta_preserves_autocast_projection_dtype(self):
        base = torch.randn(2, 3, 4, dtype=torch.bfloat16)
        delta = torch.randn(2, 3, 4, dtype=torch.float32)
        result = add_projection_delta(base, delta)
        self.assertEqual(result.dtype, torch.bfloat16)

    def test_modes_are_separated_and_legacy_names_remain_loadable(self):
        self.assertEqual(_normalise_mode("fera"), "easyvfx_faithful")
        self.assertEqual(_normalise_mode("afldm"), "afldm_spatial_faithful")
        self.assertEqual(_normalise_mode("fera_ea"), "video_temporal_extension")

        easyvfx = VideoFrequencyEquivarianceAdapter(
            dim=8, num_layers=1, mode="easyvfx_faithful", blocks="0"
        )
        afldm = VideoFrequencyEquivarianceAdapter(
            dim=8, num_layers=1, mode="afldm_spatial_faithful", blocks="0"
        )
        temporal = VideoFrequencyEquivarianceAdapter(
            dim=8, num_layers=1, mode="video_temporal_extension", blocks="0"
        )
        self.assertTrue(easyvfx.uses_frequency_experts)
        self.assertFalse(easyvfx.uses_equivariant_attention)
        self.assertFalse(afldm.uses_frequency_experts)
        self.assertTrue(afldm.uses_equivariant_attention)
        self.assertFalse(afldm.uses_temporal_extension)
        self.assertTrue(temporal.uses_frequency_experts)
        self.assertTrue(temporal.uses_temporal_extension)

    def test_frequency_indicator_separates_appearance_and_motion(self):
        indicator = FrequencyEnergyIndicator()
        static = torch.randn(2, 4, 1, 16, 16).repeat(1, 1, 5, 1, 1)
        descriptor = indicator(static)
        self.assertEqual(tuple(descriptor.shape), (2, 6))
        torch.testing.assert_close(descriptor[:, :3].sum(dim=-1), torch.ones(2))
        torch.testing.assert_close(descriptor[:, 3:], torch.zeros(2, 3))

    def test_router_matches_easyvfx_soft_routing_contract(self):
        router = SoftFrequencyRouter(num_experts=3, hidden_dim=8, temperature=0.7)
        weights = router(torch.randn(4, 6))
        self.assertIsInstance(router.net[1], torch.nn.ReLU)
        self.assertEqual(tuple(weights.shape), (4, 3))
        torch.testing.assert_close(weights.sum(dim=-1), torch.ones(4))
        self.assertTrue(torch.all(weights > 0))

    def test_checkpoint_marker_records_mode_and_blocks(self):
        adapter = VideoFrequencyEquivarianceAdapter(
            dim=8,
            num_layers=4,
            mode="video_temporal_extension",
            blocks="0,3",
            total_rank=3,
            num_experts=3,
        )
        marker_keys = [key for key in adapter.state_dict() if key.startswith("mode_markers.")]
        self.assertEqual(
            marker_keys,
            ["mode_markers.video_temporal_extension__blocks__0_3"],
        )
        with TemporaryDirectory() as directory:
            checkpoint = f"{directory}/adapter.safetensors"
            save_file(adapter.state_dict(), checkpoint)
            self.assertEqual(
                infer_video_frequency_config(checkpoint),
                {
                    "mode": "video_temporal_extension",
                    "blocks": "0,3",
                    "rank": 3,
                    "num_experts": 3,
                    "router_hidden_dim": 64,
                },
            )

    def test_pure_afldm_checkpoint_mode_is_inferable_without_experts(self):
        adapter = VideoFrequencyEquivarianceAdapter(
            dim=8,
            num_layers=4,
            mode="afldm_spatial_faithful",
            blocks="1,2",
        )
        with TemporaryDirectory() as directory:
            checkpoint = f"{directory}/adapter.safetensors"
            save_file(adapter.state_dict(), checkpoint)
            self.assertEqual(
                infer_video_frequency_config(checkpoint),
                {"mode": "afldm_spatial_faithful", "blocks": "1,2"},
            )

    def test_expert_rank_budget_and_soft_routing(self):
        experts = MultiExpertLoRA(2, 2, total_rank=5, num_experts=3)
        self.assertEqual(experts.ranks, (2, 2, 1))
        for down, up in zip(experts.down, experts.up):
            torch.nn.init.ones_(down.weight)
            torch.nn.init.ones_(up.weight)
        hidden = torch.ones(1, 1, 2)
        first = experts(hidden, torch.tensor([[1.0, 0.0, 0.0]]))
        third = experts(hidden, torch.tensor([[0.0, 0.0, 1.0]]))
        torch.testing.assert_close(first, torch.full_like(first, 4.0))
        torch.testing.assert_close(third, torch.full_like(third, 2.0))

        scaled = MultiExpertLoRA(2, 2, total_rank=5, num_experts=3, alpha=1.0)
        for down, up in zip(scaled.down, scaled.up):
            torch.nn.init.ones_(down.weight)
            torch.nn.init.ones_(up.weight)
        scaled_first = scaled(hidden, torch.tensor([[1.0, 0.0, 0.0]]))
        scaled_third = scaled(hidden, torch.tensor([[0.0, 0.0, 1.0]]))
        torch.testing.assert_close(scaled_first, torch.full_like(scaled_first, 2.0))
        torch.testing.assert_close(scaled_third, torch.full_like(scaled_third, 2.0))

    def test_bandlimited_shift_preserves_constant_and_masks_boundaries(self):
        adapter = VideoFrequencyEquivarianceAdapter(
            dim=8,
            num_layers=1,
            mode="video_temporal_extension",
            blocks="0",
            total_rank=3,
            num_experts=3,
            equivariance_weight=1.0,
        )
        video = torch.ones(2, 4, 5, 8, 8)
        shift = (0.5, -0.25, 0.25)
        shifted = adapter.bandlimited_shift(video, shift)
        torch.testing.assert_close(shifted, video, atol=1e-5, rtol=1e-5)
        mask = adapter.valid_mask(video.shape, shift, video.device)
        self.assertEqual(tuple(mask.shape), (2, 1, 5, 8, 8))
        self.assertGreater(mask.sum().item(), 0)
        self.assertLess(mask.sum().item(), mask.numel())

    def test_afldm_spatial_shift_does_not_mix_video_frames(self):
        adapter = VideoFrequencyEquivarianceAdapter(
            dim=8,
            num_layers=1,
            mode="afldm_spatial_faithful",
            blocks="0",
            equivariance_weight=1.0,
        )
        video = torch.zeros(1, 1, 3, 8, 8)
        video[:, :, 1, 3, 3] = 1
        shifted = adapter.bandlimited_shift(video, (0.0, 0.25, -0.25))
        torch.testing.assert_close(shifted[:, :, 0], torch.zeros_like(shifted[:, :, 0]))
        torch.testing.assert_close(shifted[:, :, 2], torch.zeros_like(shifted[:, :, 2]))
        self.assertGreater(shifted[:, :, 1].abs().sum().item(), 0)

    def test_afldm_shifts_spatially_aligned_video_conditions(self):
        adapter = VideoFrequencyEquivarianceAdapter(
            dim=8,
            num_layers=1,
            mode="afldm_spatial_faithful",
            blocks="0",
            equivariance_weight=1.0,
        )
        latents = torch.zeros(1, 2, 3, 8, 8)
        condition = torch.zeros_like(latents)
        latents[:, :, :, 3, 3] = 1
        condition[:, :, :, 2, 4] = 1
        shifted = adapter.shift_model_inputs(
            {"latents": latents, "vace_context": condition},
            (0.0, 0.25, -0.25),
        )
        torch.testing.assert_close(
            shifted["latents"],
            adapter.spatial_fractional_shift(latents, (0.0, 0.25, -0.25)),
        )
        torch.testing.assert_close(
            shifted["vace_context"],
            adapter.spatial_fractional_shift(condition, (0.0, 0.25, -0.25)),
        )

    def test_pre_patch_alias_filter_suppresses_spatial_checkerboard(self):
        adapter = VideoFrequencyEquivarianceAdapter(
            dim=8,
            num_layers=1,
            mode="afldm_spatial_faithful",
            blocks="0",
        )
        checker = (torch.arange(8)[:, None] + torch.arange(8)[None, :]) % 2
        video = checker.float().mul(2).sub(1).view(1, 1, 1, 8, 8)
        filtered = adapter.pre_patch_antialias(video, (1, 2, 2))
        self.assertLess(filtered.abs().max().item(), 1e-5)

    def test_equivariant_attention_uses_detached_reference_hidden_state(self):
        adapter = VideoFrequencyEquivarianceAdapter(
            dim=8,
            num_layers=1,
            mode="afldm_spatial_faithful",
            blocks="0",
        )
        hidden = torch.randn(2, 4, 8, requires_grad=True)
        adapter.attention_context(0, hidden, None, ATTENTION_STORE)
        loaded_hidden, loaded_routing = adapter.attention_context(
            0,
            torch.zeros_like(hidden),
            None,
            ATTENTION_LOAD,
        )
        self.assertFalse(loaded_hidden.requires_grad)
        self.assertIsNone(loaded_routing)
        torch.testing.assert_close(loaded_hidden, hidden.detach())

    def test_afldm_equivariance_target_keeps_gradient_by_default(self):
        adapter = VideoFrequencyEquivarianceAdapter(
            dim=8,
            num_layers=1,
            mode="afldm_spatial_faithful",
            blocks="0",
            equivariance_weight=1.0,
        )
        prediction = torch.randn(1, 2, 2, 4, 4, requires_grad=True)
        shifted = torch.randn_like(prediction, requires_grad=True)
        loss = adapter.equivariance_loss(shifted, prediction, (0.0, 0.25, 0.25))
        loss.backward()
        self.assertIsNotNone(prediction.grad)
        self.assertIsNotNone(shifted.grad)


if __name__ == "__main__":
    unittest.main()
