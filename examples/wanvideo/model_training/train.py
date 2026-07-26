import math
import torch, os, json
from diffsynth.models.adaptive_rope_runtime import attach_adaptive_rope, load_adaptive_rope_checkpoint
from diffsynth.models.easyvfx_frequency_runtime import build_easyvfx_frequency_loss
from diffsynth.models.latent_memory_runtime import attach_latent_memory, load_latent_memory_checkpoint
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from diffsynth.trainers.utils import DiffusionTrainingModule, ModelLogger, launch_training_task, wan_parser
from diffsynth.trainers.unified_dataset import UnifiedDataset, LoadVideo, ImageCropAndResize, ToAbsolutePath
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def resolve_model_init_device(model_init_device):
    if model_init_device != "cuda":
        return model_init_device
    local_rank = os.getenv("LOCAL_RANK")
    if local_rank is None or not torch.cuda.is_available():
        return model_init_device
    return f"cuda:{int(local_rank)}"

class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        tokenizer_path=None,
        model_init_device="cpu",
        trainable_models=None,
        lora_base_model=None, lora_target_modules="q,k,v,o,ffn.0,ffn.2", lora_rank=32, lora_checkpoint=None,
        latent_memory_mode="none", latent_memory_tokens=4, latent_memory_hidden_dim=0,
        latent_memory_scale=1.0, latent_memory_init_std=0.02, latent_memory_checkpoint=None,
        sa_rope_mode="none", sa_rope_hidden_dim=128, sa_rope_init_scale=0.1,
        sa_rope_fixed_temporal_scale=1.0, sa_rope_fixed_spatial_scale=1.0,
        sa_rope_min_scale=0.5, sa_rope_max_scale=1.5, sa_rope_reg_weight=0.0,
        sa_rope_head_roles_path=None, sa_rope_default_head_role="alternate",
        identity_distill_weight=0.0, first_frame_mmd_weight=0.0,
        identity_distill_max_tokens=256,
        aux_loss_preset="none",
        latent_memory_code_diversity_weight=0.0,
        latent_memory_gate_entropy_weight=0.0,
        latent_memory_gate_usage_weight=0.0,
        latent_feature_alignment_weight=0.0,
        latent_feature_alignment_margin=0.0,
        latent_feature_relation_weight=0.0,
        latent_feature_relation_margin=0.0,
        latent_feature_relation_max_tokens=256,
        easyvfx_frequency_mode="none",
        easyvfx_low_ratio=0.25,
        easyvfx_temporal_low_ratio=0.35,
        easyvfx_low_weight=0.0,
        easyvfx_high_weight=0.0,
        easyvfx_temporal_weight=0.0,
        easyvfx_descriptor_weight=0.0,
        easyvfx_adaptive_weight=0.0,
        easyvfx_adaptive_temperature=0.7,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
    ):
        super().__init__()
        # Load models
        print("[WanTrainingModule] parse_model_configs start", flush=True)
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, enable_fp8_training=False)
        print("[WanTrainingModule] parse_model_configs done", flush=True)
        tokenizer_config = ModelConfig(path=tokenizer_path) if tokenizer_path is not None else None
        resolved_model_init_device = resolve_model_init_device(model_init_device)
        print(f"[WanTrainingModule] resolved model_init_device={resolved_model_init_device}", flush=True)
        print("[WanTrainingModule] from_pretrained start", flush=True)
        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=resolved_model_init_device,
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
        )
        print("[WanTrainingModule] from_pretrained done", flush=True)
        
        # Training mode
        print("[WanTrainingModule] switch_pipe_to_training_mode start", flush=True)
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint=lora_checkpoint,
            enable_fp8_training=False,
        )
        print("[WanTrainingModule] switch_pipe_to_training_mode done", flush=True)
        adapter, prefixes = attach_latent_memory(
            self.pipe,
            mode=latent_memory_mode,
            num_tokens=latent_memory_tokens,
            hidden_dim=latent_memory_hidden_dim,
            scale=latent_memory_scale,
            init_std=latent_memory_init_std,
            log_prefix="[WanTrainingModule] latent_memory",
        )
        load_latent_memory_checkpoint(
            adapter,
            latent_memory_checkpoint or lora_checkpoint,
            prefixes,
            required=latent_memory_checkpoint is not None,
            log_prefix="[WanTrainingModule] latent_memory",
        )
        sa_rope_adapter = attach_adaptive_rope(
            self.pipe,
            mode=sa_rope_mode,
            hidden_dim=sa_rope_hidden_dim,
            init_scale=sa_rope_init_scale,
            fixed_temporal_scale=sa_rope_fixed_temporal_scale,
            fixed_spatial_scale=sa_rope_fixed_spatial_scale,
            min_scale=sa_rope_min_scale,
            max_scale=sa_rope_max_scale,
            head_roles_path=sa_rope_head_roles_path,
            default_head_role=sa_rope_default_head_role,
            log_prefix="[WanTrainingModule] sa_rope",
        )
        load_adaptive_rope_checkpoint(
            sa_rope_adapter,
            lora_checkpoint,
            required=False,
            log_prefix="[WanTrainingModule] sa_rope",
        )
        if sa_rope_adapter is not None:
            sa_rope_adapter.hidden_trace_max_tokens = int(identity_distill_max_tokens)
        
        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        self.aux_loss_config = self._build_aux_loss_config(
            aux_loss_preset=aux_loss_preset,
            latent_memory_code_diversity_weight=latent_memory_code_diversity_weight,
            latent_memory_gate_entropy_weight=latent_memory_gate_entropy_weight,
            latent_memory_gate_usage_weight=latent_memory_gate_usage_weight,
            latent_feature_alignment_weight=latent_feature_alignment_weight,
            latent_feature_alignment_margin=latent_feature_alignment_margin,
            latent_feature_relation_weight=latent_feature_relation_weight,
            latent_feature_relation_margin=latent_feature_relation_margin,
            latent_feature_relation_max_tokens=latent_feature_relation_max_tokens,
            sa_rope_reg_weight=sa_rope_reg_weight,
            identity_distill_weight=identity_distill_weight,
            first_frame_mmd_weight=first_frame_mmd_weight,
            identity_distill_max_tokens=identity_distill_max_tokens,
        )
        if self._has_aux_loss():
            print(f"[WanTrainingModule] aux_loss_config={self.aux_loss_config}", flush=True)
        self.easyvfx_frequency_loss = build_easyvfx_frequency_loss(
            mode=easyvfx_frequency_mode,
            low_ratio=easyvfx_low_ratio,
            temporal_low_ratio=easyvfx_temporal_low_ratio,
            low_weight=easyvfx_low_weight,
            high_weight=easyvfx_high_weight,
            temporal_weight=easyvfx_temporal_weight,
            descriptor_weight=easyvfx_descriptor_weight,
            adaptive_weight=easyvfx_adaptive_weight,
            adaptive_temperature=easyvfx_adaptive_temperature,
        )
        if self.easyvfx_frequency_loss is not None:
            print(
                "[WanTrainingModule] easyvfx_frequency="
                f"mode={easyvfx_frequency_mode}, low_ratio={easyvfx_low_ratio}, "
                f"temporal_low_ratio={easyvfx_temporal_low_ratio}, "
                f"weights=(low={easyvfx_low_weight}, high={easyvfx_high_weight}, "
                f"temporal={easyvfx_temporal_weight}, descriptor={easyvfx_descriptor_weight}, "
                f"adaptive={easyvfx_adaptive_weight})",
                flush=True,
            )
        
    @staticmethod
    def _build_aux_loss_config(**kwargs):
        config = dict(kwargs)
        preset = str(config.pop("aux_loss_preset", "none") or "none")
        if preset not in {"none", "codebook_gate", "latent_align", "all"}:
            raise ValueError(f"Unknown --aux_loss_preset: {preset}")
        if preset in {"codebook_gate", "all"}:
            config["latent_memory_code_diversity_weight"] = config["latent_memory_code_diversity_weight"] or 1e-3
            config["latent_memory_gate_entropy_weight"] = config["latent_memory_gate_entropy_weight"] or 1e-4
            config["latent_memory_gate_usage_weight"] = config["latent_memory_gate_usage_weight"] or 1e-3
        if preset in {"latent_align", "all"}:
            config["latent_feature_alignment_weight"] = config["latent_feature_alignment_weight"] or 1e-2
            config["latent_feature_relation_weight"] = config["latent_feature_relation_weight"] or 1e-3
        config["aux_loss_preset"] = preset
        config["latent_feature_relation_max_tokens"] = max(2, int(config["latent_feature_relation_max_tokens"]))
        return config

    def _has_aux_loss(self):
        keys = (
            "latent_memory_code_diversity_weight",
            "latent_memory_gate_entropy_weight",
            "latent_memory_gate_usage_weight",
            "latent_feature_alignment_weight",
            "latent_feature_relation_weight",
            "sa_rope_reg_weight",
            "identity_distill_weight",
            "first_frame_mmd_weight",
        )
        if any(float(self.aux_loss_config.get(key, 0.0) or 0.0) != 0.0 for key in keys):
            return True
        return getattr(self, "easyvfx_frequency_loss", None) is not None

    def _sa_rope_regularization_loss(self):
        adapter = getattr(self.pipe, "sa_rope_adapter", None)
        if adapter is None:
            return None
        return adapter.regularization_loss()

    def _latent_memory_adapters(self):
        adapters = []
        text_adapter = getattr(self.pipe, "latent_memory_text", None)
        if text_adapter is not None:
            adapters.append(text_adapter)
        vace = getattr(self.pipe, "vace", None)
        if vace is not None:
            for name in ("latent_memory_context", "latent_memory_hint"):
                adapter = getattr(vace, name, None)
                if adapter is not None:
                    adapters.append(adapter)
        return adapters

    def _latent_memory_code_diversity_loss(self):
        losses = []
        for adapter in self._latent_memory_adapters():
            memory = getattr(adapter, "memory", None)
            if memory is None:
                continue
            memory = memory.float()
            if memory.dim() == 2:
                memory = memory.unsqueeze(0)
            elif memory.dim() > 3:
                memory = memory.reshape(-1, memory.shape[-2], memory.shape[-1])
            if memory.shape[-2] < 2:
                continue
            normalized = torch.nn.functional.normalize(memory, dim=-1)
            sim = torch.matmul(normalized, normalized.transpose(-1, -2))
            eye = torch.eye(sim.shape[-1], device=sim.device, dtype=sim.dtype).unsqueeze(0)
            offdiag = sim * (1.0 - eye)
            denom = sim.shape[0] * sim.shape[-1] * (sim.shape[-1] - 1)
            losses.append(offdiag.pow(2).sum() / max(1, denom))
        if not losses:
            return None
        return torch.stack(losses).mean()

    def _latent_memory_route_probs(self):
        probs = []
        for adapter in self._latent_memory_adapters():
            history = getattr(adapter, "route_probs_history", None)
            if history:
                probs.extend(item.float() for item in history if item is not None)
                continue
            last = getattr(adapter, "last_route_probs", None)
            if last is not None:
                probs.append(last.float())
        return probs

    def _latent_memory_gate_entropy_loss(self):
        losses = []
        for probs in self._latent_memory_route_probs():
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            entropy = -(probs.clamp_min(1e-6) * probs.clamp_min(1e-6).log()).sum(dim=-1)
            if probs.shape[-1] > 1:
                entropy = entropy / math.log(probs.shape[-1])
            losses.append(entropy.mean())
        if not losses:
            return None
        return torch.stack(losses).mean()

    def _latent_memory_gate_usage_loss(self):
        losses = []
        for probs in self._latent_memory_route_probs():
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            usage = probs.mean(dim=0)
            uniform = torch.full_like(usage, 1.0 / usage.numel())
            losses.append((usage.clamp_min(1e-6) * (usage.clamp_min(1e-6) / uniform).log()).sum())
        if not losses:
            return None
        return torch.stack(losses).mean()

    @staticmethod
    def _latent_tokens(tensor):
        tensor = tensor.float()
        if tensor.dim() == 5:
            return tensor.permute(0, 2, 3, 4, 1).reshape(tensor.shape[0], -1, tensor.shape[1])
        if tensor.dim() == 4:
            return tensor.permute(1, 2, 3, 0).reshape(1, -1, tensor.shape[0])
        if tensor.dim() == 3:
            return tensor
        if tensor.dim() == 2:
            return tensor.unsqueeze(0)
        return tensor.reshape(1, -1, 1)

    def _latent_feature_alignment_loss(self, pred_x0, target_x0):
        pred = self._latent_tokens(pred_x0)
        target = self._latent_tokens(target_x0).to(device=pred.device, dtype=pred.dtype)
        token_count = min(pred.shape[1], target.shape[1])
        pred = pred[:, :token_count]
        target = target[:, :token_count]
        cosine = torch.nn.functional.cosine_similarity(pred, target, dim=-1)
        margin = float(self.aux_loss_config.get("latent_feature_alignment_margin", 0.0) or 0.0)
        return torch.relu(1.0 - margin - cosine).mean()

    def _latent_feature_relation_loss(self, pred_x0, target_x0):
        pred = self._latent_tokens(pred_x0)
        target = self._latent_tokens(target_x0).to(device=pred.device, dtype=pred.dtype)
        token_count = min(pred.shape[1], target.shape[1])
        max_tokens = int(self.aux_loss_config.get("latent_feature_relation_max_tokens", 256))
        sample_count = min(token_count, max_tokens)
        if sample_count < 2:
            return None
        indices = torch.linspace(0, token_count - 1, steps=sample_count, device=pred.device).long()
        pred = torch.nn.functional.normalize(pred[:, indices], dim=-1)
        target = torch.nn.functional.normalize(target[:, indices], dim=-1)
        pred_rel = torch.matmul(pred, pred.transpose(-1, -2))
        target_rel = torch.matmul(target, target.transpose(-1, -2))
        margin = float(self.aux_loss_config.get("latent_feature_relation_margin", 0.0) or 0.0)
        return torch.relu((pred_rel - target_rel).abs() - margin).mean()

    def _trace_frame_tokens(self, trace_entry):
        if "tokens" in trace_entry:
            return trace_entry["tokens"].float()
        hidden = trace_entry.get("hidden")
        grid = trace_entry.get("grid_size")
        if hidden is None or grid is None:
            return None
        f, h, w = [int(x) for x in grid]
        spatial = h * w
        token_count = min(hidden.shape[1], f * spatial)
        if token_count < f:
            return None
        spatial = token_count // f
        tokens = hidden[:, :f * spatial].reshape(hidden.shape[0], f, spatial, hidden.shape[-1])
        max_tokens = int(self.aux_loss_config.get("identity_distill_max_tokens", 256) or 256)
        sample_count = min(spatial, max_tokens)
        if sample_count < spatial:
            idx = torch.linspace(0, spatial - 1, sample_count, device=hidden.device).long()
            tokens = tokens[:, :, idx]
        return tokens.float()

    def _identity_relation_distill_loss(self, outputs):
        student = outputs.get("student_hidden_trace") or {}
        teacher = outputs.get("identity_hidden_trace") or {}
        losses = []
        for block_id in sorted(set(student) & set(teacher)):
            s_tokens = self._trace_frame_tokens(student[block_id])
            t_tokens = self._trace_frame_tokens(teacher[block_id])
            if s_tokens is None or t_tokens is None:
                continue
            s_frame = torch.nn.functional.normalize(s_tokens.mean(dim=2), dim=-1)
            t_frame = torch.nn.functional.normalize(t_tokens.mean(dim=2), dim=-1)
            s_rel = torch.matmul(s_frame, s_frame.transpose(-1, -2))
            t_rel = torch.matmul(t_frame, t_frame.transpose(-1, -2))
            losses.append(torch.nn.functional.mse_loss(s_rel, t_rel.detach()))
        if not losses:
            return None
        return torch.stack(losses).mean()

    @staticmethod
    def _mmd_rbf(x, y, gamma=None):
        if x.numel() == 0 or y.numel() == 0:
            return None
        x = x.reshape(-1, x.shape[-1])
        y = y.reshape(-1, y.shape[-1])
        if gamma is None:
            gamma = 1.0 / max(1, x.shape[-1])
        xx = torch.cdist(x, x).pow(2)
        yy = torch.cdist(y, y).pow(2)
        xy = torch.cdist(x, y).pow(2)
        return torch.exp(-gamma * xx).mean() + torch.exp(-gamma * yy).mean() - 2 * torch.exp(-gamma * xy).mean()

    def _first_frame_mmd_loss(self, outputs):
        student = outputs.get("student_hidden_trace") or {}
        teacher = outputs.get("identity_hidden_trace") or {}
        losses = []
        for block_id in sorted(set(student) & set(teacher)):
            s_tokens = self._trace_frame_tokens(student[block_id])
            t_tokens = self._trace_frame_tokens(teacher[block_id])
            if s_tokens is None or t_tokens is None or s_tokens.shape[1] < 2:
                continue
            s_tokens = torch.nn.functional.normalize(s_tokens, dim=-1)
            t_tokens = torch.nn.functional.normalize(t_tokens, dim=-1)
            for frame_id in range(1, s_tokens.shape[1]):
                s_rel = torch.matmul(s_tokens[:, 0], s_tokens[:, frame_id].transpose(-1, -2))
                t_rel = torch.matmul(t_tokens[:, 0], t_tokens[:, frame_id].transpose(-1, -2))
                loss = self._mmd_rbf(s_rel, t_rel.detach())
                if loss is not None:
                    losses.append(loss)
        if not losses:
            return None
        return torch.stack(losses).mean()

    def _auxiliary_loss(self, outputs):
        terms = []
        weighted_terms = []

        def add_term(name, value, weight):
            weight = float(weight or 0.0)
            if weight == 0.0 or value is None:
                return
            terms.append(f"{name}={float(value.detach().item()):.6f}*{weight:g}")
            weighted_terms.append(value * weight)

        add_term(
            "code_diversity",
            self._latent_memory_code_diversity_loss(),
            self.aux_loss_config.get("latent_memory_code_diversity_weight"),
        )
        add_term(
            "gate_entropy",
            self._latent_memory_gate_entropy_loss(),
            self.aux_loss_config.get("latent_memory_gate_entropy_weight"),
        )
        add_term(
            "gate_usage",
            self._latent_memory_gate_usage_loss(),
            self.aux_loss_config.get("latent_memory_gate_usage_weight"),
        )
        add_term(
            "latent_align",
            self._latent_feature_alignment_loss(outputs["pred_x0"], outputs["target_x0"]),
            self.aux_loss_config.get("latent_feature_alignment_weight"),
        )
        add_term(
            "latent_relation",
            self._latent_feature_relation_loss(outputs["pred_x0"], outputs["target_x0"]),
            self.aux_loss_config.get("latent_feature_relation_weight"),
        )
        add_term(
            "sa_rope_reg",
            self._sa_rope_regularization_loss(),
            self.aux_loss_config.get("sa_rope_reg_weight"),
        )
        add_term(
            "identity_relation",
            self._identity_relation_distill_loss(outputs),
            self.aux_loss_config.get("identity_distill_weight"),
        )
        add_term(
            "first_frame_mmd",
            self._first_frame_mmd_loss(outputs),
            self.aux_loss_config.get("first_frame_mmd_weight"),
        )
        easyvfx_loss, easyvfx_metrics = self._easyvfx_frequency_aux_loss(outputs)
        add_term("easyvfx_freq", easyvfx_loss, 1.0)
        for metric_name, metric_value in easyvfx_metrics.items():
            terms.append(f"easyvfx_{metric_name}={float(metric_value.detach().item()):.6f}")
        if not weighted_terms:
            return outputs["loss"], ""
        aux_loss = torch.stack(weighted_terms).sum()
        return outputs["loss"] + aux_loss, "; ".join(terms)

    def _easyvfx_frequency_aux_loss(self, outputs):
        module = getattr(self, "easyvfx_frequency_loss", None)
        if module is None:
            return None, {}
        return module(outputs["pred_x0"], outputs["target_x0"])
        
    def forward_preprocess(self, data):
        # CFG-sensitive parameters
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        
        # CFG-unsensitive parameters
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_video": data["video"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }
        
        # Extra inputs
        for extra_input in self.extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data["video"][0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][-1]
            elif extra_input == "reference_image" or extra_input == "vace_reference_image":
                inputs_shared[extra_input] = data[extra_input][0]
            else:
                inputs_shared[extra_input] = data[extra_input]
        
        # Pipeline units will automatically process the input parameters.
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)
        return {**inputs_shared, **inputs_posi}
    
    
    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.forward_preprocess(data)
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        if not self._has_aux_loss():
            loss = self.pipe.training_loss(**models, **inputs)
            return loss
        needs_identity = (
            float(self.aux_loss_config.get("identity_distill_weight", 0.0) or 0.0) != 0.0
            or float(self.aux_loss_config.get("first_frame_mmd_weight", 0.0) or 0.0) != 0.0
        )
        outputs = self.pipe.training_loss(
            **models,
            **inputs,
            return_outputs=True,
            identity_propagation=needs_identity,
        )
        loss, aux_summary = self._auxiliary_loss(outputs)
        if aux_summary:
            self._last_aux_loss_summary = aux_summary
        return loss


if __name__ == "__main__":
    parser = wan_parser()
    args = parser.parse_args()
    print("[train.py] building dataset", flush=True)
    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=UnifiedDataset.default_video_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
            num_frames=args.num_frames,
            time_division_factor=4,
            time_division_remainder=1,
        ),
        special_operator_map={
            "animate_face_video": ToAbsolutePath(args.dataset_base_path) >> LoadVideo(args.num_frames, 4, 1, frame_processor=ImageCropAndResize(512, 512, None, 16, 16))
        }
    )
    print(f"[train.py] dataset ready len={len(dataset)}", flush=True)
    print("[train.py] building model", flush=True)
    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        model_init_device=args.model_init_device,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        latent_memory_mode=args.latent_memory_mode,
        latent_memory_tokens=args.latent_memory_tokens,
        latent_memory_hidden_dim=args.latent_memory_hidden_dim,
        latent_memory_scale=args.latent_memory_scale,
        latent_memory_init_std=args.latent_memory_init_std,
        latent_memory_checkpoint=args.latent_memory_checkpoint,
        sa_rope_mode=args.sa_rope_mode,
        sa_rope_hidden_dim=args.sa_rope_hidden_dim,
        sa_rope_init_scale=args.sa_rope_init_scale,
        sa_rope_fixed_temporal_scale=args.sa_rope_fixed_temporal_scale,
        sa_rope_fixed_spatial_scale=args.sa_rope_fixed_spatial_scale,
        sa_rope_min_scale=args.sa_rope_min_scale,
        sa_rope_max_scale=args.sa_rope_max_scale,
        sa_rope_reg_weight=args.sa_rope_reg_weight,
        sa_rope_head_roles_path=args.sa_rope_head_roles_path,
        sa_rope_default_head_role=args.sa_rope_default_head_role,
        identity_distill_weight=args.identity_distill_weight,
        first_frame_mmd_weight=args.first_frame_mmd_weight,
        identity_distill_max_tokens=args.identity_distill_max_tokens,
        aux_loss_preset=args.aux_loss_preset,
        latent_memory_code_diversity_weight=args.latent_memory_code_diversity_weight,
        latent_memory_gate_entropy_weight=args.latent_memory_gate_entropy_weight,
        latent_memory_gate_usage_weight=args.latent_memory_gate_usage_weight,
        latent_feature_alignment_weight=args.latent_feature_alignment_weight,
        latent_feature_alignment_margin=args.latent_feature_alignment_margin,
        latent_feature_relation_weight=args.latent_feature_relation_weight,
        latent_feature_relation_margin=args.latent_feature_relation_margin,
        latent_feature_relation_max_tokens=args.latent_feature_relation_max_tokens,
        easyvfx_frequency_mode=args.easyvfx_frequency_mode,
        easyvfx_low_ratio=args.easyvfx_low_ratio,
        easyvfx_temporal_low_ratio=args.easyvfx_temporal_low_ratio,
        easyvfx_low_weight=args.easyvfx_low_weight,
        easyvfx_high_weight=args.easyvfx_high_weight,
        easyvfx_temporal_weight=args.easyvfx_temporal_weight,
        easyvfx_descriptor_weight=args.easyvfx_descriptor_weight,
        easyvfx_adaptive_weight=args.easyvfx_adaptive_weight,
        easyvfx_adaptive_temperature=args.easyvfx_adaptive_temperature,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
    )
    print("[train.py] model ready", flush=True)
    if args.sa_rope_trace_head_roles_output:
        adapter = getattr(model.pipe, "sa_rope_adapter", None)
        if adapter is None:
            raise ValueError("--sa_rope_trace_head_roles_output requires --sa_rope_mode != none")
        adapter.trace_head_roles = True
        adapter.trace_max_spatial_tokens = int(args.sa_rope_trace_max_spatial_tokens)
        adapter.reset_attention_role_votes()
        sample_count = min(int(args.sa_rope_trace_head_roles_samples), len(dataset))
        print(f"[train.py] tracing SA-RoPE head roles samples={sample_count}", flush=True)
        model.eval()
        with torch.no_grad():
            for idx in range(sample_count):
                model(dataset[idx])
        adapter.save_head_roles_json(args.sa_rope_trace_head_roles_output)
        print(f"[train.py] saved SA-RoPE head roles: {args.sa_rope_trace_head_roles_output}", flush=True)
        raise SystemExit(0)
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt
    )
    print("[train.py] launching training task", flush=True)
    launch_training_task(dataset, model, model_logger, args=args)
    print("[train.py] training task finished", flush=True)
