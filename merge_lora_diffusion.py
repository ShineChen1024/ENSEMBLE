"""Merge a LoRA adapter into Qwen-Image-Edit diffusion transformer weights."""

import argparse
import os

import torch
from diffusers import QwenImageEditPlusPipeline


def parse_args():
    p = argparse.ArgumentParser(description="Merge LoRA into Qwen-Image-Edit diffusion model")
    p.add_argument("--base", required=True, help="Base Qwen-Image-Edit model path")
    p.add_argument("--lora", required=True, help="LoRA adapter directory")
    p.add_argument("--out", required=True, help="Output directory for merged pipeline")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--adapter-name", default="ensemble_lora")
    p.add_argument("--lora-scale", type=float, default=1.0)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    print(f"[INFO] Loading base pipeline: {args.base}")
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        args.base, text_encoder=None, torch_dtype=torch_dtype,
    )

    print(f"[INFO] Loading LoRA adapter: {args.lora}")
    pipe.load_lora_weights(args.lora, adapter_name=args.adapter_name)
    pipe.set_adapters(args.adapter_name, args.lora_scale)

    print("[INFO] Fusing LoRA into transformer weights ...")
    pipe.fuse_lora(adapter_names=[args.adapter_name])
    pipe.unload_lora_weights()

    print(f"[INFO] Saving merged pipeline to: {args.out}")
    pipe.save_pretrained(args.out, safe_serialization=True)

    print(f"\n[OK] Done. Load with:\n  QwenImageEditPlusPipeline.from_pretrained('{args.out}')")


if __name__ == "__main__":
    main()
