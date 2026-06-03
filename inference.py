"""ENSEMBLE: single-image inference script."""

import argparse
import math
from pathlib import Path

import torch
from PIL import Image, ImageOps
from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

SYSTEM_PROMPT = (
    "Describe the key features of the clothing item in the input image, including color, shape, "
    "material, style, and visual details, then act as a fashion stylist to recommend a complete "
    "outfit. Explain the styling choice naturally and provide a clear outfit description suitable "
    "for image generation."
)
USER_PROMPT = "请根据这件单品推荐完整穿搭，先写<reason>搭配思路</reason>，再给出穿搭描述。"
IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"


def parse_args():
    p = argparse.ArgumentParser(description="ENSEMBLE single-image inference")
    p.add_argument("--image", required=True, help="Input garment image path")
    p.add_argument("--model-path", required=True, help="Path to merged ENSEMBLE model (contains text_encoder, transformer, etc.)")
    p.add_argument("--output", default="output.png", help="Output image path")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--cfg-scale", type=float, default=4.0)
    return p.parse_args()


def resize_condition(image, max_short_side=960):
    short_side = math.floor(max_short_side / 32) * 32
    w, h = image.size
    if w <= h:
        width = short_side
        height = round((short_side * h / w) / 32) * 32
    else:
        height = short_side
        width = round((short_side * w / h) / 32) * 32
    return image.resize((width, height), Image.Resampling.LANCZOS)


@torch.no_grad()
def stylist_recommend(text_encoder, processor, tokenizer, image_path, device):
    """Run QwenVL stylist: generate recommendation text and extract hidden states."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": f"file://{image_path}"},
            {"type": "text", "text": USER_PROMPT},
        ]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(device)

    outputs = text_encoder.generate(
        **inputs, max_new_tokens=512, do_sample=False,
        return_dict_in_generate=True, output_hidden_states=True,
    )

    input_len = inputs["input_ids"].shape[1]
    assistant_text = processor.batch_decode(
        outputs.sequences[:, input_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )[0].strip()

    last_hidden = torch.cat([step[-1] for step in outputs.hidden_states], dim=1)

    # Drop system prefix to align with training template
    prefix = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n"
    drop_idx = len(tokenizer(prefix, add_special_tokens=False).input_ids)
    prompt_embeds = last_hidden[:, drop_idx:].to(dtype=torch.bfloat16, device=device)
    prompt_mask = torch.ones(prompt_embeds.shape[:2], dtype=torch.long, device=device)

    return assistant_text, prompt_embeds, prompt_mask


def main():
    args = parse_args()
    device = "cuda"

    from diffusers import QwenImageEditPlusPipeline

    model_path = Path(args.model_path)
    text_encoder_path = str(model_path / "text_encoder")

    # Load text encoder (stylist VLM)
    print(f"Loading text encoder: {text_encoder_path}")
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        text_encoder_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device).eval()

    processor = AutoProcessor.from_pretrained(text_encoder_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(text_encoder_path, trust_remote_code=True)
    processor.image_min_pixels = 32 * 32
    processor.image_max_pixels = 512 * 512
    if hasattr(processor, "image_processor"):
        processor.image_processor.size = {"shortest_edge": 32 * 32, "longest_edge": 512 * 512}

    # Load diffusion pipeline
    print(f"Loading diffusion model: {args.model_path}")
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        args.model_path, text_encoder=None, processor=processor,
        tokenizer=tokenizer, torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload()

    # Prepare input image
    image_path = str(Path(args.image).resolve())
    input_image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    condition_image = resize_condition(input_image)

    # Step 1: Stylist recommendation + embedding extraction
    print("Generating outfit recommendation...")
    text_encoder.to(device)
    assistant_text, prompt_embeds, prompt_mask = stylist_recommend(
        text_encoder, processor, tokenizer, image_path, device,
    )
    text_encoder.to("cpu")
    torch.cuda.empty_cache()

    print(f"Recommendation: {assistant_text}")

    # Step 2: Generate outfit image
    print("Generating outfit image...")
    generator = torch.Generator(device=device).manual_seed(args.seed)
    neg_embeds = torch.zeros_like(prompt_embeds) if args.cfg_scale > 1 else None
    neg_mask = torch.ones_like(prompt_mask) if args.cfg_scale > 1 else None

    result = pipe(
        image=condition_image,
        prompt=None,
        prompt_embeds=prompt_embeds,
        prompt_embeds_mask=prompt_mask,
        negative_prompt=None,
        negative_prompt_embeds=neg_embeds,
        negative_prompt_embeds_mask=neg_mask,
        true_cfg_scale=args.cfg_scale,
        num_inference_steps=args.steps,
        generator=generator,
        max_sequence_length=512,
    )

    result.images[0].save(args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
