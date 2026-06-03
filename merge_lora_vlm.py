"""Merge a LoRA adapter into Qwen2.5-VL (stylist VLM) base weights."""

import argparse
import os

import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def parse_args():
    p = argparse.ArgumentParser(description="Merge LoRA into Qwen2.5-VL base model")
    p.add_argument("--base", required=True, help="Base Qwen2.5-VL model path")
    p.add_argument("--lora", required=True, help="LoRA adapter directory")
    p.add_argument("--out", required=True, help="Output directory for merged model")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--device-map", default="cpu", choices=["cpu", "auto"])
    p.add_argument("--max-shard-size", default="5GB")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    print(f"[INFO] Loading base model: {args.base}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base, torch_dtype=torch_dtype,
        device_map=None if args.device_map == "cpu" else "auto",
    )

    print(f"[INFO] Loading LoRA adapter: {args.lora}")
    model = PeftModel.from_pretrained(model, args.lora, is_trainable=False)

    print("[INFO] Merging LoRA into base weights ...")
    merged = model.merge_and_unload(progressbar=True, safe_merge=True)
    merged.to(torch_dtype)

    print(f"[INFO] Saving merged model to: {args.out}")
    merged.save_pretrained(args.out, safe_serialization=True, max_shard_size=args.max_shard_size)

    print("[INFO] Saving processor ...")
    processor = AutoProcessor.from_pretrained(args.base)
    processor.save_pretrained(args.out)

    print(f"\n[OK] Done. Load with:\n  Qwen2_5_VLForConditionalGeneration.from_pretrained('{args.out}')")


if __name__ == "__main__":
    main()
