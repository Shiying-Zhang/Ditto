import argparse
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diffsynth import save_video, VideoData
from diffsynth.models.latent_memory_runtime import attach_latent_memory, load_latent_memory_checkpoint
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig


DEFAULT_NEGATIVE_PROMPT = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"


def _existing_weight(path: Path) -> bool:
    return path.exists() and path.is_file() and not path.name.endswith((".fdmdownload", ".crdownload"))


def _local_model_configs(checkpoint_path: str) -> tuple[list, ModelConfig]:
    model_dir = Path(checkpoint_path).expanduser().resolve()
    transformer_paths = [str(path) for path in sorted(model_dir.glob("diffusion_pytorch_model*.safetensors"))]
    text_encoder_path = model_dir / "models_t5_umt5-xxl-enc-bf16.pth"
    vae_path = model_dir / "Wan2.1_VAE.pth"
    tokenizer_dir = model_dir / "google" / "umt5-xxl"
    if not transformer_paths or not _existing_weight(text_encoder_path) or not _existing_weight(vae_path) or not tokenizer_dir.exists():
        raise RuntimeError(
            f"Incomplete local checkpoint at {model_dir}. Expected diffusion shards, text encoder, VAE, and google/umt5-xxl tokenizer files."
        )
    model_configs = [
        ModelConfig(path=transformer_paths, offload_device="cpu"),
        ModelConfig(path=[str(text_encoder_path)], offload_device="cpu"),
        ModelConfig(path=[str(vae_path)], offload_device="cpu"),
    ]
    tokenizer_config = ModelConfig(path=str(tokenizer_dir))
    return model_configs, tokenizer_config


def _parse_multi_values(raw_values, raw_alphas, default_alpha: float) -> tuple[list[str], list[float]]:
    lora_paths: list[str] = []
    lora_alphas: list[float] = []
    for raw in raw_values or []:
        for item in str(raw).split(","):
            item = item.strip()
            if item:
                lora_paths.append(item)
    if not lora_paths:
        return [], []
    if not raw_alphas:
        return lora_paths, [default_alpha] * len(lora_paths)
    if len(raw_alphas) == 1 and len(lora_paths) > 1:
        return lora_paths, [float(raw_alphas[0])] * len(lora_paths)
    parsed_alphas = [float(item) for item in raw_alphas]
    if len(parsed_alphas) != len(lora_paths):
        raise ValueError("The number of lora_alpha values must match the number of lora_path values, or provide a single shared alpha.")
    return lora_paths, parsed_alphas


def _resolve_latent_memory_checkpoint(args, lora_paths: list[str]) -> str:
    if args.latent_memory_checkpoint:
        return args.latent_memory_checkpoint
    if args.latent_memory_mode == "none":
        return ""
    if not lora_paths:
        return ""
    if len(lora_paths) > 1:
        print(
            "Warning: multiple LoRA checkpoints were provided without --latent_memory_checkpoint. "
            "Using the first LoRA checkpoint for latent memory weights.",
            flush=True,
        )
    return lora_paths[0]


def main(args):

    device = f"cuda:{args.device_id}"

    if args.checkpoint_path:
        model_configs, tokenizer_config = _local_model_configs(args.checkpoint_path)
        pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device,
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
            redirect_common_files=False,
        )
    else:
        pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device,
            model_configs=[
                ModelConfig(model_id="Wan-AI/Wan2.1-VACE-14B", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu"),
                ModelConfig(model_id="Wan-AI/Wan2.1-VACE-14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
                ModelConfig(model_id="Wan-AI/Wan2.1-VACE-14B", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
            ],
        )

    lora_paths, lora_alphas = _parse_multi_values(args.lora_path, args.lora_alpha, args.default_lora_alpha)
    latent_memory_checkpoint = _resolve_latent_memory_checkpoint(args, lora_paths)
    if args.latent_memory_mode != "none":
        adapter, prefixes = attach_latent_memory(
            pipe,
            mode=args.latent_memory_mode,
            num_tokens=args.latent_memory_tokens,
            hidden_dim=args.latent_memory_hidden_dim,
            scale=args.latent_memory_scale,
            init_std=args.latent_memory_init_std,
            log_prefix="[infer_ditto] latent_memory",
        )
        if not latent_memory_checkpoint:
            raise RuntimeError(
                "latent memory inference requires --latent_memory_checkpoint, or at least one --lora_path "
                "whose checkpoint also contains latent memory weights."
            )
        load_latent_memory_checkpoint(
            adapter,
            latent_memory_checkpoint,
            prefixes,
            required=True,
            log_prefix="[infer_ditto] latent_memory",
        )
    for lora_path, alpha in zip(lora_paths, lora_alphas):
        print(f"Loading Ditto LoRA model: {lora_path} (alpha={alpha})")
        if not os.path.exists(lora_path):
            print(f"Error: LoRA file not found at {lora_path}")
            return
        pipe.load_lora(pipe.vace, lora_path, alpha=alpha)

    pipe.enable_vram_management()

    print(f"Loading input video: {args.input_video}")
    if not os.path.exists(args.input_video):
        print(f"Error: Input video file not found at {args.input_video}")
        return
        
    input_video_path = args.control_video or args.input_video
    video = VideoData(input_video_path, height=args.height, width=args.width)
    
    num_frames = min(args.num_frames, len(video))
    if num_frames != args.num_frames:
        print(f"Warning: Requested number of frames ({args.num_frames}) exceeds total video frames ({len(video)}). Using {num_frames} frames instead.")
        
    video = [video[i] for i in range(num_frames)]
    
    reference_image = None
    if args.reference_image:
        reference_image = Image.open(args.reference_image).convert("RGB").resize((args.width, args.height))

    video = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        vace_video=video,
        vace_reference_image=reference_image,
        num_frames=num_frames,
        num_inference_steps=args.num_inference_steps,
        cfg_scale=args.cfg_scale,
        seed=args.seed,
        tiled=True,
    )

    output_dir = os.path.dirname(args.output_video)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    save_video(video, args.output_video, fps=args.fps, quality=args.quality)
    metadata = {
        "checkpoint_path": args.checkpoint_path or "Wan-AI/Wan2.1-VACE-14B",
        "input_video": args.input_video,
        "control_video": args.control_video or "",
        "reference_image": args.reference_image or "",
        "output_video": args.output_video,
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "lora_paths": lora_paths,
        "lora_alphas": lora_alphas,
        "latent_memory_mode": args.latent_memory_mode,
        "latent_memory_tokens": args.latent_memory_tokens,
        "latent_memory_hidden_dim": args.latent_memory_hidden_dim,
        "latent_memory_scale": args.latent_memory_scale,
        "latent_memory_init_std": args.latent_memory_init_std,
        "latent_memory_checkpoint": latent_memory_checkpoint,
        "resolution": [args.width, args.height],
        "num_frames": num_frames,
        "fps": args.fps,
        "seed": args.seed,
        "num_inference_steps": args.num_inference_steps,
        "cfg_scale": args.cfg_scale,
    }
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="InstructV2V Pipeline.")

    parser.add_argument("--input_video", type=str, required=True, help="Path to the input video file.")
    parser.add_argument("--output_video", type=str, required=True, help="Path to save the output video file.")
    parser.add_argument("--checkpoint_path", type=str, default="", help="Optional local Wan2.1-VACE-14B checkpoint directory.")
    parser.add_argument("--control_video", type=str, default="", help="Optional control/depth video path. Defaults to input_video when omitted.")
    parser.add_argument("--reference_image", type=str, default="", help="Optional reference image path.")
    parser.add_argument("--lora_path", action="append", default=[], help="Optional LoRA path. Can be provided multiple times or as a comma-separated list.")
    parser.add_argument("--device_id", type=int, default=0, help="The ID of the CUDA device to use (e.g., 0, 1, 2).")
    parser.add_argument("--prompt", type=str, required=True, help="The positive prompt describing the target style.")
    parser.add_argument("--negative_prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT, help="Negative prompt.")
    parser.add_argument("--height", type=int, default=480, help="The height to use for video processing.")
    parser.add_argument("--width", type=int, default=832, help="The width to use for video processing.")
    parser.add_argument("--num_frames", type=int, default=73, help="The number of video frames to process.")
    parser.add_argument("--num_inference_steps", type=int, default=20, help="Number of diffusion sampling steps.")
    parser.add_argument("--cfg_scale", type=float, default=1.0, help="Classifier-free guidance scale.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for reproducible results.")

    parser.add_argument("--lora_alpha", action="append", default=[], help="Optional alpha for each LoRA. Provide one value to share across all LoRAs.")
    parser.add_argument("--default_lora_alpha", type=float, default=1.0, help="Fallback alpha when lora_alpha is omitted.")
    parser.add_argument("--latent_memory_mode", choices=["none", "text", "vace_context", "vace_hint"], default="none", help="Attach latent memory adapter during inference.")
    parser.add_argument("--latent_memory_tokens", type=int, default=4, help="Number of latent memory tokens.")
    parser.add_argument("--latent_memory_hidden_dim", type=int, default=0, help="Hidden dimension for latent memory conditioning MLP. 0 uses an automatic value.")
    parser.add_argument("--latent_memory_scale", type=float, default=1.0, help="Scale applied to latent memory output.")
    parser.add_argument("--latent_memory_init_std", type=float, default=0.02, help="Initialization std for latent memory parameters.")
    parser.add_argument("--latent_memory_checkpoint", type=str, default="", help="Checkpoint containing latent memory weights. Defaults to the first LoRA checkpoint when latent_memory_mode is enabled.")
    parser.add_argument("--fps", type=int, default=20, help="Frames per second (FPS) for the output video.")
    parser.add_argument("--quality", type=int, default=5, help="Quality of the output video (CRF value, lower is better).")

    args = parser.parse_args()
    main(args)
