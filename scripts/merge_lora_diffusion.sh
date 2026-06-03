#!/bin/bash
source /home/work/MMSearch/cwf/miniconda/bin/activate search-train

python /home/work/MMSearch/cwf/search/ENSEMBLE/merge_lora_diffusion.py \
    --base /home/work/MMSearch/cwf/search/model/Qwen-Image-Edit-2511 \
    --lora /home/work/MMSearch/svg-shared-model-new-copy/ai-search/outfit/finetune/mytheresa_qwen_edit_lora_attn_add_imgmlp_r64_gbs16_steps1720_lr5e-5_cond512-960 \
    --out /home/work/MMSearch/cwf/search/model/Qwen-Image-Edit-2511-merged \
    --dtype bf16
