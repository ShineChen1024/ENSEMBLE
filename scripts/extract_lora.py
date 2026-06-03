#!/usr/bin/env python3
import sys
import os
from safetensors.torch import load_file, save_file

CHECKPOINTS = [
    "/home/work/MMSearch/cwf/ENSEMBLE/qwen_image_edit_lora_qkv/checkpoint-500",
    "/home/work/MMSearch/cwf/ENSEMBLE/qwen_image_edit_lora_qkv/checkpoint-1000",
    "/home/work/MMSearch/cwf/ENSEMBLE/qwen_image_edit_lora_qkv/checkpoint-1500",
]

def rename_key(k):
    k = k.replace(".lora_A.default.", ".lora_A.")
    k = k.replace(".lora_B.default.", ".lora_B.")
    k = "transformer." + k
    return k

for ckpt_dir in CHECKPOINTS:
    src = os.path.join(ckpt_dir, "model.safetensors")
    dst = os.path.join(ckpt_dir, "pytorch_lora_weights.safetensors")

    print(f"Processing {ckpt_dir} ...")
    state_dict = load_file(src)

    lora_dict = {}
    for k, v in state_dict.items():
        if "lora" in k.lower():
            lora_dict[rename_key(k)] = v

    print(f"  Extracted {len(lora_dict)} lora keys")
    save_file(lora_dict, dst)
    print(f"  Saved to {dst}")

print("Done!")
