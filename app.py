"""ENSEMBLE: Gradio web UI for interactive outfit styling."""

import math
import tempfile
from pathlib import Path

import gradio as gr
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

# Global model references (loaded once at startup)
text_encoder = None
processor = None
tokenizer = None
pipe = None


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
def stylist_recommend(image_path, device):
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
    prefix = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n"
    drop_idx = len(tokenizer(prefix, add_special_tokens=False).input_ids)
    prompt_embeds = last_hidden[:, drop_idx:].to(dtype=torch.bfloat16, device=device)
    prompt_mask = torch.ones(prompt_embeds.shape[:2], dtype=torch.long, device=device)

    return assistant_text, prompt_embeds, prompt_mask


def generate(input_image, seed, steps, cfg_scale):
    if input_image is None:
        raise gr.Error("Please upload a garment image.")

    device = "cuda"
    image = ImageOps.exif_transpose(Image.fromarray(input_image)).convert("RGB")

    # Save to temp file for QwenVL (it needs a file path)
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    image.save(tmp.name)

    condition_image = resize_condition(image)

    # Step 1: Stylist recommendation
    text_encoder.to(device)
    assistant_text, prompt_embeds, prompt_mask = stylist_recommend(tmp.name, device)
    text_encoder.to("cpu")
    torch.cuda.empty_cache()

    # Step 2: Diffusion generation
    generator = torch.Generator(device=device).manual_seed(seed)
    neg_embeds = torch.zeros_like(prompt_embeds) if cfg_scale > 1 else None
    neg_mask = torch.ones_like(prompt_mask) if cfg_scale > 1 else None

    result = pipe(
        image=condition_image,
        prompt=None,
        prompt_embeds=prompt_embeds,
        prompt_embeds_mask=prompt_mask,
        negative_prompt=None,
        negative_prompt_embeds=neg_embeds,
        negative_prompt_embeds_mask=neg_mask,
        true_cfg_scale=cfg_scale,
        num_inference_steps=steps,
        generator=generator,
        max_sequence_length=512,
    )

    # Parse reason and description from assistant text
    reason = ""
    description = assistant_text
    if "<reason>" in assistant_text and "</reason>" in assistant_text:
        reason = assistant_text.split("<reason>")[1].split("</reason>")[0].strip()
        description = assistant_text.split("</reason>")[-1].strip()

    styling_output = f"**Styling Rationale:** {reason}\n\n**Outfit Description:** {description}"
    return result.images[0], styling_output


def load_models(model_path):
    global text_encoder, processor, tokenizer, pipe

    from diffusers import QwenImageEditPlusPipeline

    text_encoder_path = str(Path(model_path) / "text_encoder")

    print(f"Loading text encoder: {text_encoder_path}")
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        text_encoder_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to("cuda").eval()

    processor = AutoProcessor.from_pretrained(text_encoder_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(text_encoder_path, trust_remote_code=True)
    processor.image_min_pixels = 32 * 32
    processor.image_max_pixels = 512 * 512
    if hasattr(processor, "image_processor"):
        processor.image_processor.size = {"shortest_edge": 32 * 32, "longest_edge": 512 * 512}

    print(f"Loading diffusion model: {model_path}")
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        model_path, text_encoder=None, processor=processor,
        tokenizer=tokenizer, torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload()

    text_encoder.to("cpu")
    torch.cuda.empty_cache()
    print("Models loaded.")


def build_ui():
    with gr.Blocks(title="ENSEMBLE", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# ENSEMBLE\n> Upload a garment image, get a styled outfit recommendation and generated look.")

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="Input Garment", type="numpy")
                with gr.Row():
                    seed = gr.Slider(0, 9999, value=0, step=1, label="Seed")
                    steps = gr.Slider(10, 50, value=30, step=1, label="Steps")
                cfg_scale = gr.Slider(1.0, 10.0, value=4.0, step=0.5, label="CFG Scale")
                run_btn = gr.Button("Generate Outfit", variant="primary")

            with gr.Column():
                output_image = gr.Image(label="Generated Outfit")
                styling_text = gr.Markdown(label="Styling Recommendation")

        run_btn.click(fn=generate, inputs=[input_image, seed, steps, cfg_scale], outputs=[output_image, styling_text])

    return demo


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="ENSEMBLE Gradio UI")
    p.add_argument("--model-path", required=True, help="Path to merged ENSEMBLE model")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true")
    args = p.parse_args()

    load_models(args.model_path)
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)
