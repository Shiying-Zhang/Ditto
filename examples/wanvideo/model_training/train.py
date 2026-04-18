import torch, os, json
from diffsynth import load_state_dict
from diffsynth.models.latent_memory import (
    LatentMemoryContextAdapter,
    LatentMemoryHintAdapter,
    LatentMemoryTextAdapter,
    extract_latent_memory_state_dict,
)
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


def attach_latent_memory(
    pipe,
    mode="none",
    num_tokens=4,
    hidden_dim=0,
    scale=1.0,
    init_std=0.02,
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
    print(f"[WanTrainingModule] latent_memory attached mode={mode} trainable_params={param_count}", flush=True)
    return adapter, prefixes


def load_latent_memory_checkpoint(adapter, checkpoint, prefixes, required=False):
    if adapter is None or checkpoint is None:
        return
    state_dict = load_state_dict(checkpoint)
    adapter_state = adapter.state_dict()
    latent_state = extract_latent_memory_state_dict(state_dict, adapter_state.keys(), prefixes)
    if not latent_state:
        if required:
            print(f"[WanTrainingModule] warning: no latent_memory keys found in {checkpoint}", flush=True)
        return
    load_result = adapter.load_state_dict(latent_state, strict=False)
    print(f"[WanTrainingModule] latent_memory checkpoint loaded: {checkpoint}, total {len(latent_state)} keys", flush=True)
    if len(load_result[0]) > 0:
        print(f"[WanTrainingModule] warning: missing latent_memory keys: {load_result[0]}", flush=True)
    if len(load_result[1]) > 0:
        print(f"[WanTrainingModule] warning: unexpected latent_memory keys: {load_result[1]}", flush=True)



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
        )
        load_latent_memory_checkpoint(
            adapter,
            latent_memory_checkpoint or lora_checkpoint,
            prefixes,
            required=latent_memory_checkpoint is not None,
        )
        
        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        
        
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
        loss = self.pipe.training_loss(**models, **inputs)
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
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
    )
    print("[train.py] model ready", flush=True)
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt
    )
    print("[train.py] launching training task", flush=True)
    launch_training_task(dataset, model, model_logger, args=args)
    print("[train.py] training task finished", flush=True)
