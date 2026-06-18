#!/usr/bin/env python3
"""Train Qwen-Image-Edit LoRA on Mytheresa outfit pairs with the merged stylist QwenVL encoder."""

import argparse
import gc
import json
import math
import os
import random
import sys
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration


DIFFUSERS_SRC = Path("/home/work/MMSearch/cwf/search/diffusers/src")
QWEN_IMAGE_EDIT_MODEL = "/home/work/MMSearch/cwf/search/model/Qwen-Image-Edit-2511"
MERGED_QWENVL_MODEL = "/home/work/MMSearch/cwf/search/model/mytheresa_stylist_qwen25vl_7b_merged"
TRAIN_JSONL = Path(
    "/home/work/MMSearch/cwf/search/outfit/"
    "mytheresa_recommender_reason_zh.handwritten_1_9330.qc_clean.all_revised.jsonl"
)
OUTPUT_DIR = Path("/home/work/MMSearch/svg-shared-model-new-copy/ai-search/outfit/finetune/qwen_image_edit_mytheresa_lora")
TEXT_ENCODER_MIN_PIXELS = 32 * 32
TEXT_ENCODER_MAX_PIXELS = 512 * 512
VAE_IMAGE_AREA = 1024 * 1024
CONDITION_SHORT_SIDE_MIN = 512
CONDITION_SHORT_SIDE_MAX = 960

SYSTEM_PROMPT = (
    "Describe the key features of the clothing item in the input image, including color, shape, "
    "material, style, and visual details, then act as a fashion stylist to recommend a complete "
    "outfit. Explain the styling choice naturally and provide a clear outfit description suitable "
    "for image generation."
)
USER_PROMPT = "请根据这件单品推荐完整穿搭，先写<reason>搭配思路</reason>，再给出穿搭描述。"

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=Path, default=TRAIN_JSONL)
    parser.add_argument("--pretrained-model", default=QWEN_IMAGE_EDIT_MODEL)
    parser.add_argument("--text-encoder-model", default=MERGED_QWENVL_MODEL)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--target-image-area", type=int, default=VAE_IMAGE_AREA)
    parser.add_argument("--condition-short-side-min", type=int, default=CONDITION_SHORT_SIDE_MIN)
    parser.add_argument("--condition-short-side-max", type=int, default=CONDITION_SHORT_SIDE_MAX)
    parser.add_argument("--text-encoder-min-pixels", type=int, default=TEXT_ENCODER_MIN_PIXELS)
    parser.add_argument("--text-encoder-max-pixels", type=int, default=TEXT_ENCODER_MAX_PIXELS)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lr-warmup-steps", type=int, default=100)
    parser.add_argument("--max-train-steps", type=int, default=2000)
    parser.add_argument("--num-train-epochs", type=int, default=1)
    parser.add_argument("--checkpointing-steps", type=int, default=500, help="Save checkpoint every N steps. 0 disables intermediate checkpoints.")
    parser.add_argument("--checkpointing-mode", choices=["resume", "lora", "both", "full"], default="resume")
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--lora-layers", default="to_k,to_q,to_v,to_out.0")
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--dataloader-num-workers", type=int, default=2)
    parser.add_argument("--report-to", default="tensorboard")
    parser.add_argument("--weighting-scheme", choices=["none", "sigma_sqrt", "logit_normal", "mode", "cosmap"], default="none")
    parser.add_argument("--logit-mean", type=float, default=0.0)
    parser.add_argument("--logit-std", type=float, default=1.0)
    parser.add_argument("--mode-scale", type=float, default=1.29)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    return parser.parse_args()


def add_diffusers_to_path():
    sys.path.insert(0, str(DIFFUSERS_SRC))


def read_assistant_text(row):
    messages = row.get("messages") or []
    for message in messages:
        if message.get("role") == "assistant" and isinstance(message.get("content"), str):
            return message["content"].strip()

    reason = row.get("recommendation_reason")
    outfit = row.get("final_outfit_description")
    if reason and outfit:
        return f"<reason>{reason}</reason>\n{outfit}"
    raise ValueError("row has no assistant answer or recommendation_reason/final_outfit_description")


def image_path(row, *keys):
    for key in keys:
        value = row.get(key)
        if value:
            return value
    raise ValueError(f"missing image key among {keys}")


def calculate_dimensions(target_area, ratio):
    width = math.sqrt(target_area * ratio)
    height = width / ratio
    return round(width / 32) * 32, round(height / 32) * 32


def resize_to_area(image, target_area):
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = calculate_dimensions(target_area, image.width / image.height)
    return image.resize((width, height), Image.Resampling.LANCZOS)


def resize_by_random_short_side(image, min_short_side, max_short_side):
    image = ImageOps.exif_transpose(image).convert("RGB")
    min_units = math.ceil(min_short_side / 32)
    max_units = math.floor(max_short_side / 32)
    if min_units > max_units:
        raise ValueError("condition short side range must contain at least one multiple of 32")

    short_side = random.randint(min_units, max_units) * 32
    if image.width <= image.height:
        width = short_side
        height = round((short_side * image.height / image.width) / 32) * 32
    else:
        height = short_side
        width = round((short_side * image.width / image.height) / 32) * 32
    return image.resize((width, height), Image.Resampling.LANCZOS)


def prepare_text_condition_image(image):
    return ImageOps.exif_transpose(image).convert("RGB")


class MytheresaQwenImageEditDataset(Dataset):
    def __init__(self, jsonl_path, target_image_area, condition_short_side_min, condition_short_side_max, max_samples=None):
        self.target_image_area = target_image_area
        self.condition_short_side_min = condition_short_side_min
        self.condition_short_side_max = condition_short_side_max
        self.rows = []
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                row = json.loads(line)
                try:
                    condition_path = Path(image_path(row, "item_product_full_path")).resolve()
                    target_path = Path(image_path(row, "outfit_look_path")).resolve()
                    assistant_text = read_assistant_text(row)
                except Exception as exc:
                    raise ValueError(f"{jsonl_path}:{line_no}: invalid row: {exc}") from exc
                if not condition_path.exists() or not target_path.exists():
                    continue
                self.rows.append(
                    {
                        "condition_path": str(condition_path),
                        "target_path": str(target_path),
                        "assistant_text": assistant_text,
                    }
                )
                if max_samples and len(self.rows) >= max_samples:
                    break
        if not self.rows:
            raise ValueError(f"no valid rows found in {jsonl_path}")

        self.to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        condition = Image.open(row["condition_path"])
        target = Image.open(row["target_path"])
        condition_text_image = prepare_text_condition_image(condition)
        condition_vae_image = resize_by_random_short_side(
            condition,
            self.condition_short_side_min,
            self.condition_short_side_max,
        )
        target_image = resize_to_area(target, self.target_image_area)
        return {
            "condition_text_image": condition_text_image,
            "condition_vae_pixels": self.to_tensor(condition_vae_image).unsqueeze(1),
            "target_pixels": self.to_tensor(target_image).unsqueeze(1),
            "assistant_text": row["assistant_text"],
            "condition_path": row["condition_path"],
            "target_path": row["target_path"],
        }


def collate_fn(examples):
    return {
        "condition_text_images": [example["condition_text_image"] for example in examples],
        "condition_vae_pixels": torch.stack([example["condition_vae_pixels"] for example in examples]),
        "target_pixels": torch.stack([example["target_pixels"] for example in examples]),
        "assistant_texts": [example["assistant_text"] for example in examples],
        "condition_paths": [example["condition_path"] for example in examples],
        "target_paths": [example["target_path"] for example in examples],
    }


def build_qwenvl_chat_text(assistant_text, user_prompt=USER_PROMPT):
    return (
        "<|im_start|>system\n"
        f"{SYSTEM_PROMPT}"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "<|vision_start|><|image_pad|><|vision_end|>"
        f"{user_prompt}"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        f"{assistant_text}"
        "<|im_end|>\n"
    )


def user_content_start_idx(tokenizer):
    prefix = (
        "<|im_start|>system\n"
        f"{SYSTEM_PROMPT}"
        "<|im_end|>\n"
        "<|im_start|>user\n"
    )
    return len(tokenizer(prefix, add_special_tokens=False).input_ids)


@torch.no_grad()
def encode_prompts(text_encoder, processor, tokenizer, condition_images, assistant_texts, device, dtype):
    texts = [build_qwenvl_chat_text(assistant_text) for assistant_text in assistant_texts]

    inputs = processor(
        text=texts,
        images=condition_images,
        padding=True,
        return_tensors="pt",
    ).to(device)
    outputs = text_encoder(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        pixel_values=inputs.get("pixel_values"),
        image_grid_thw=inputs.get("image_grid_thw"),
        output_hidden_states=True,
    )
    hidden_states = outputs.hidden_states[-1]
    valid_lengths = inputs["attention_mask"].bool().sum(dim=1).tolist()
    split_hidden_states = torch.split(hidden_states[inputs["attention_mask"].bool()], valid_lengths, dim=0)

    drop_idx = user_content_start_idx(tokenizer)
    split_hidden_states = [hidden[drop_idx:] for hidden in split_hidden_states]
    max_len = max(hidden.shape[0] for hidden in split_hidden_states)
    embeds = torch.stack(
        [torch.cat([hidden, hidden.new_zeros(max_len - hidden.shape[0], hidden.shape[1])]) for hidden in split_hidden_states]
    )
    mask = torch.stack(
        [
            torch.cat(
                [
                    torch.ones(hidden.shape[0], dtype=torch.long, device=hidden.device),
                    torch.zeros(max_len - hidden.shape[0], dtype=torch.long, device=hidden.device),
                ]
            )
            for hidden in split_hidden_states
        ]
    )
    if mask.all():
        mask = None
    return embeds.to(device=device, dtype=dtype), mask


def set_qwenvl_image_pixels(processor, min_pixels, max_pixels):
    processor.image_min_pixels = min_pixels
    processor.image_max_pixels = max_pixels
    if hasattr(processor, "image_processor"):
        processor.image_processor.size = {"shortest_edge": min_pixels, "longest_edge": max_pixels}


def tracker_config(args):
    config = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            config[key] = str(value)
        elif value is None:
            config[key] = "None"
        else:
            config[key] = value
    return config


def retrieve_latents(encoder_output, sample=True):
    if hasattr(encoder_output, "latent_dist"):
        return encoder_output.latent_dist.sample() if sample else encoder_output.latent_dist.mode()
    if hasattr(encoder_output, "latents"):
        return encoder_output.latents
    raise AttributeError("could not retrieve latents from VAE output")


def torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def trainable_state_dict(model):
    return {name: param.detach().cpu() for name, param in model.named_parameters() if param.requires_grad}


def normalize_lora_key(key):
    return (
        key.replace("._fsdp_wrapped_module", "")
        .replace(".lora_A.default.weight", ".lora_A.weight")
        .replace(".lora_B.default.weight", ".lora_B.weight")
    )


def lora_shape_map(model):
    shapes = {}
    for module_name, module in model.named_modules():
        module_name = module_name.replace("._fsdp_wrapped_module", "")
        if hasattr(module, "lora_A"):
            for adapter_name, lora_layer in module.lora_A.items():
                suffix = ".lora_A.weight" if adapter_name == "default" else f".lora_A.{adapter_name}.weight"
                shapes[f"{module_name}{suffix}"] = (lora_layer.out_features, lora_layer.in_features)
        if hasattr(module, "lora_B"):
            for adapter_name, lora_layer in module.lora_B.items():
                suffix = ".lora_B.weight" if adapter_name == "default" else f".lora_B.{adapter_name}.weight"
                shapes[f"{module_name}{suffix}"] = (lora_layer.out_features, lora_layer.in_features)
    return shapes


def load_lora_from_rank_checkpoints(checkpoint_dir, num_processes, model):
    shapes = lora_shape_map(model)
    lora_state = {}
    for rank in range(num_processes):
        rank_state_path = checkpoint_dir / f"rank_{rank}_state.pt"
        if not rank_state_path.exists():
            continue
        rank_state = torch_load(rank_state_path)
        for key, value in rank_state.get("trainable_params", {}).items():
            key = normalize_lora_key(key)
            if key not in shapes or value.numel() == 0:
                continue
            shape = shapes[key]
            if value.numel() != math.prod(shape):
                logger.warning(f"skip LoRA tensor with unexpected shape: {key} shard={tuple(value.shape)} expected={shape}")
                continue
            lora_state[key] = value.reshape(shape).contiguous()
        del rank_state
    missing = sorted(set(shapes) - set(lora_state))
    if missing:
        raise RuntimeError(f"missing {len(missing)} LoRA tensors while exporting checkpoint; first missing: {missing[:5]}")
    return lora_state


def load_trainable_state_dict(model, state_dict):
    missing = []
    with torch.no_grad():
        params = dict(model.named_parameters())
        for name, value in state_dict.items():
            if name not in params:
                missing.append(name)
                continue
            target = params[name]
            value = value.to(device=target.device, dtype=target.dtype)
            if target.shape == value.shape:
                target.copy_(value)
            elif target.numel() == value.numel():
                target.view(-1).copy_(value.view(-1))
            else:
                missing.append(name)
    return missing


def save_lora_training_checkpoint(
    accelerator,
    transformer,
    optimizer,
    lr_scheduler,
    args,
    checkpoint_dir,
    global_step,
    save_lora_weights_fn,
):
    if accelerator.is_main_process:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    unwrapped_transformer = accelerator.unwrap_model(transformer)
    if accelerator.is_main_process:
        torch.save(
            {
                "global_step": global_step,
                "checkpointing_mode": args.checkpointing_mode,
                "args": vars(args),
            },
            checkpoint_dir / "training_state.pt",
        )

    rank_state = {
        "trainable_params": trainable_state_dict(unwrapped_transformer),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        rank_state["cuda_rng_state"] = torch.cuda.get_rng_state()
    torch.save(rank_state, checkpoint_dir / f"rank_{accelerator.process_index}_state.pt")
    accelerator.wait_for_everyone()

    if args.checkpointing_mode in {"lora", "both"}:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if accelerator.is_main_process:
            lora_state = load_lora_from_rank_checkpoints(checkpoint_dir, accelerator.num_processes, unwrapped_transformer)
            torch.save(lora_state, checkpoint_dir / "lora_state.pt")
            save_lora_weights_fn(
                save_directory=checkpoint_dir,
                transformer_lora_layers=lora_state,
            )
        accelerator.wait_for_everyone()


def load_lora_training_checkpoint(accelerator, transformer, optimizer, lr_scheduler, checkpoint_dir):
    lora_state_path = checkpoint_dir / "lora_state.pt"
    training_state_path = checkpoint_dir / "training_state.pt"
    if not training_state_path.exists():
        raise FileNotFoundError(f"training state not found in checkpoint: {training_state_path}")

    unwrapped_transformer = accelerator.unwrap_model(transformer)
    if lora_state_path.exists():
        lora_state = torch_load(lora_state_path)
        incompatible_keys = set_peft_model_state_dict(unwrapped_transformer, lora_state, adapter_name="default")
        if accelerator.is_main_process and incompatible_keys is not None:
            logger.info(f"loaded LoRA state from {lora_state_path}; incompatible_keys={incompatible_keys}")

    training_state = torch_load(training_state_path)
    rank_state_path = checkpoint_dir / f"rank_{accelerator.process_index}_state.pt"
    if rank_state_path.exists():
        rank_state = torch_load(rank_state_path)
        if "trainable_params" in rank_state:
            missing = load_trainable_state_dict(unwrapped_transformer, rank_state["trainable_params"])
            if missing and accelerator.is_main_process:
                logger.warning(f"some trainable params from checkpoint were not found: {missing[:10]}")
        optimizer.load_state_dict(rank_state["optimizer"])
        lr_scheduler.load_state_dict(rank_state["lr_scheduler"])
        if "torch_rng_state" in rank_state:
            torch.set_rng_state(rank_state["torch_rng_state"])
        if torch.cuda.is_available() and "cuda_rng_state" in rank_state:
            torch.cuda.set_rng_state(rank_state["cuda_rng_state"])
    elif accelerator.is_main_process:
        logger.warning(f"rank state not found, optimizer/scheduler not restored: {rank_state_path}")

    accelerator.wait_for_everyone()
    return int(training_state.get("global_step", 0))


def save_final_lora_weights(accelerator, transformer, output_dir, save_lora_weights_fn):
    tmp_dir = output_dir / "final_lora_export_tmp"
    if accelerator.is_main_process:
        tmp_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    unwrapped_transformer = accelerator.unwrap_model(transformer)
    torch.save(
        {"trainable_params": trainable_state_dict(unwrapped_transformer)},
        tmp_dir / f"rank_{accelerator.process_index}_state.pt",
    )
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        lora_state = load_lora_from_rank_checkpoints(tmp_dir, accelerator.num_processes, unwrapped_transformer)
        torch.save(lora_state, output_dir / "lora_state.pt")
        save_lora_weights_fn(
            save_directory=output_dir,
            transformer_lora_layers=lora_state,
        )
    accelerator.wait_for_everyone()


def main():
    args = parse_args()
    if args.rank <= 0:
        raise ValueError(f"--rank must be a positive integer, got {args.rank}")
    if args.lora_alpha <= 0:
        raise ValueError(f"--lora-alpha must be a positive integer, got {args.lora_alpha}")
    if args.condition_short_side_min <= 0 or args.condition_short_side_max <= 0:
        raise ValueError("condition short side range must be positive")
    if args.condition_short_side_min > args.condition_short_side_max:
        raise ValueError("condition-short-side-min must be <= condition-short-side-max")

    add_diffusers_to_path()
    from diffusers import AutoencoderKLQwenImage, FlowMatchEulerDiscreteScheduler, QwenImageEditPlusPipeline, QwenImageTransformer2DModel
    from diffusers.optimization import get_scheduler
    from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=ProjectConfiguration(project_dir=args.output_dir, logging_dir=args.output_dir / "logs"),
    )
    logging_kwargs = {"main_process_only": False}
    logger.info(accelerator.state, **logging_kwargs)

    if args.seed is not None:
        set_seed(args.seed)
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if accelerator.is_main_process:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = MytheresaQwenImageEditDataset(
        args.jsonl,
        target_image_area=args.target_image_area,
        condition_short_side_min=args.condition_short_side_min,
        condition_short_side_max=args.condition_short_side_max,
        max_samples=args.max_train_samples,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
    )

    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.pretrained_model, subfolder="scheduler", shift=3.0)
    vae = AutoencoderKLQwenImage.from_pretrained(args.pretrained_model, subfolder="vae", torch_dtype=weight_dtype)
    transformer = QwenImageTransformer2DModel.from_pretrained(
        args.pretrained_model,
        subfolder="transformer",
        torch_dtype=weight_dtype,
    )
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.text_encoder_model,
        torch_dtype=weight_dtype,
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(args.text_encoder_model, trust_remote_code=True)
    set_qwenvl_image_pixels(processor, args.text_encoder_min_pixels, args.text_encoder_max_pixels)
    tokenizer = AutoTokenizer.from_pretrained(args.text_encoder_model, trust_remote_code=True)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    transformer.requires_grad_(False)
    target_modules = [layer.strip() for layer in args.lora_layers.split(",") if layer.strip()]
    transformer.add_adapter(
        LoraConfig(
            r=args.rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
    )
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)

    trainable_params = [p for p in transformer.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=1e-4, eps=1e-8)

    overrode_max_train_steps = args.max_train_steps is None
    num_update_steps_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    lr_scheduler = get_scheduler(
        "constant_with_warmup",
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    transformer, optimizer, dataloader, lr_scheduler = accelerator.prepare(transformer, optimizer, dataloader, lr_scheduler)
    num_update_steps_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    resume_global_step = 0
    if args.resume_from_checkpoint:
        resume_global_step = load_lora_training_checkpoint(
            accelerator,
            transformer,
            optimizer,
            lr_scheduler,
            args.resume_from_checkpoint,
        )
        if accelerator.is_main_process:
            logger.info(f"resumed lightweight LoRA checkpoint from {args.resume_from_checkpoint} at step {resume_global_step}")
    accelerator.init_trackers("mytheresa-qwen-image-edit-lora", config=tracker_config(args))

    vae_scale_factor = 2 ** len(vae.temperal_downsample)
    latent_channels = transformer.module.config.in_channels // 4 if hasattr(transformer, "module") else transformer.config.in_channels // 4
    latents_mean = torch.tensor(vae.config.latents_mean).view(1, latent_channels, 1, 1, 1).to(accelerator.device, weight_dtype)
    latents_std = torch.tensor(vae.config.latents_std).view(1, latent_channels, 1, 1, 1).to(accelerator.device, weight_dtype)

    def get_sigmas(timesteps, n_dim, dtype):
        sigmas = noise_scheduler.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps.to(accelerator.device)]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_main_process, desc="Steps")
    global_step = resume_global_step
    if global_step > 0:
        progress_bar.update(min(global_step, args.max_train_steps))

    for _ in range(args.num_train_epochs):
        transformer.train()
        for batch in dataloader:
            with accelerator.accumulate(transformer):
                with torch.no_grad():
                    prompt_embeds, prompt_embeds_mask = encode_prompts(
                        text_encoder,
                        processor,
                        tokenizer,
                        batch["condition_text_images"],
                        batch["assistant_texts"],
                        accelerator.device,
                        weight_dtype,
                    )

                    target_pixels = batch["target_pixels"].to(accelerator.device, dtype=weight_dtype)
                    condition_pixels = batch["condition_vae_pixels"].to(accelerator.device, dtype=weight_dtype)
                    target_latents = retrieve_latents(vae.encode(target_pixels), sample=True)
                    condition_latents = retrieve_latents(vae.encode(condition_pixels), sample=False)
                    target_latents = ((target_latents - latents_mean) / latents_std).to(weight_dtype)
                    condition_latents = ((condition_latents - latents_mean) / latents_std).to(weight_dtype)

                noise = torch.randn_like(target_latents)
                batch_size = target_latents.shape[0]
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=batch_size,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                indices = (u * noise_scheduler.config.num_train_timesteps).long()
                timesteps = noise_scheduler.timesteps[indices].to(device=target_latents.device)
                sigmas = get_sigmas(timesteps, n_dim=target_latents.ndim, dtype=target_latents.dtype)
                noisy_target_latents = (1.0 - sigmas) * target_latents + sigmas * noise

                packed_noisy_target = QwenImageEditPlusPipeline._pack_latents(
                    noisy_target_latents.permute(0, 2, 1, 3, 4),
                    batch_size=batch_size,
                    num_channels_latents=latent_channels,
                    height=target_latents.shape[3],
                    width=target_latents.shape[4],
                )
                packed_condition = QwenImageEditPlusPipeline._pack_latents(
                    condition_latents.permute(0, 2, 1, 3, 4),
                    batch_size=batch_size,
                    num_channels_latents=latent_channels,
                    height=condition_latents.shape[3],
                    width=condition_latents.shape[4],
                )
                latent_model_input = torch.cat([packed_noisy_target, packed_condition], dim=1)
                target_pixel_height = target_pixels.shape[-2]
                target_pixel_width = target_pixels.shape[-1]
                img_shapes = [
                    [
                        (1, target_latents.shape[3] // 2, target_latents.shape[4] // 2),
                        (1, condition_latents.shape[3] // 2, condition_latents.shape[4] // 2),
                    ]
                ] * batch_size

                model_pred = transformer(
                    hidden_states=latent_model_input,
                    timestep=timesteps / 1000,
                    encoder_hidden_states=prompt_embeds,
                    encoder_hidden_states_mask=prompt_embeds_mask,
                    img_shapes=img_shapes,
                    return_dict=False,
                )[0]
                model_pred = model_pred[:, : packed_noisy_target.shape[1]]
                model_pred = QwenImageEditPlusPipeline._unpack_latents(
                    model_pred,
                    target_pixel_height,
                    target_pixel_width,
                    vae_scale_factor,
                )
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
                target = noise - target_latents
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(batch_size, -1),
                    dim=1,
                ).mean()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}, step=global_step)
                progress_bar.set_postfix(loss=loss.detach().item(), lr=lr_scheduler.get_last_lr()[0])

                if args.checkpointing_steps > 0 and global_step % args.checkpointing_steps == 0:
                    checkpoint_dir = args.output_dir / f"checkpoint-{global_step}"
                    if args.checkpointing_mode == "full":
                        accelerator.save_state(checkpoint_dir)
                    else:
                        offloaded_frozen_modules = args.checkpointing_mode in {"lora", "both"} and torch.cuda.is_available()
                        if offloaded_frozen_modules:
                            text_encoder.to("cpu")
                            vae.to("cpu")
                            gc.collect()
                            torch.cuda.empty_cache()
                            accelerator.wait_for_everyone()
                        try:
                            save_lora_training_checkpoint(
                                accelerator,
                                transformer,
                                optimizer,
                                lr_scheduler,
                                args,
                                checkpoint_dir,
                                global_step,
                                QwenImageEditPlusPipeline.save_lora_weights,
                            )
                        finally:
                            if offloaded_frozen_modules:
                                vae.to(accelerator.device, dtype=weight_dtype)
                                text_encoder.to(accelerator.device, dtype=weight_dtype)
                    if accelerator.is_main_process:
                        logger.info(f"saved {args.checkpointing_mode} checkpoint to {checkpoint_dir}")

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    if torch.cuda.is_available():
        text_encoder.to("cpu")
        vae.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()
        accelerator.wait_for_everyone()
    save_final_lora_weights(
        accelerator,
        transformer,
        args.output_dir,
        QwenImageEditPlusPipeline.save_lora_weights,
    )
    if accelerator.is_main_process:
        with (args.output_dir / "training_config.json").open("w", encoding="utf-8") as f:
            json.dump(vars(args), f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"saved LoRA weights to {args.output_dir}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
