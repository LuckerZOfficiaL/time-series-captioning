from functools import lru_cache
import json
import os
from pathlib import Path
import re
import time

import requests
from PIL import Image
from qwen_omni_utils import process_mm_info
import torch
from transformers import MllamaForConditionalGeneration, AutoProcessor

from .inference_utils import run_all_tasks
from ..constants import PROJECT_ROOT, DATA_ROOT

MODEL_PATH = "meta-llama/Llama-3.2-11B-Vision-Instruct"
FINETUNED_PATH = "/shared/tsqa/finetuned_models/llama"
DATA_DIR = f"{PROJECT_ROOT}/easy_subsample"
OUT_DIR = f"{DATA_ROOT}/llama_inference_easy"


@lru_cache
def _load_batch_llama_model(model_name, device):
    torch.manual_seed(314)
    model = MllamaForConditionalGeneration.from_pretrained(
                MODEL_PATH, 
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    processor.tokenizer.padding_side = 'left'
    return model, processor


def eval_batch_llama(prompts, image_files, device, use_image): 
    print(f"use_image={use_image}")
    for i, p in enumerate(prompts):
        if "<image" in p:
            prompts[i] = re.sub(r"<image_(\d+)>", r"<|image|>", p)

    model, processor = _load_batch_llama_model(MODEL_PATH, device)
    if use_image:
        if "<|image|>" in prompts[0]:
            # Multi-image prompts
            prompts = ["<|begin_of_text|>" + p for p in prompts]
        else:
            # Single-image prompts
            prompts = ["<|image|><|begin_of_text|>" + p for p in prompts]
    else:
        prompts = ["<|begin_of_text|>" + p for p in prompts]

    if use_image:
        images = [[Image.open(fn) for fn in curr_images] for curr_images in image_files] 
    else:
        images = None
    try:
        inputs = processor(text=prompts, images=images, padding=True, return_tensors="pt")
    except:
        import pdb; pdb.set_trace()

    inputs = inputs.to(model.device).to(model.dtype)

    # Batch Inference
    stime = time.time()
    with torch.no_grad():
        text_ids = model.generate(**inputs, max_new_tokens=20, temperature=0.3, do_sample=True)
    prompt_len = inputs["input_ids"].shape[1]
    completions = text_ids[:, prompt_len:]
    print(f"RUNTIME on {device}: {time.time() - stime:.2f} seconds")
    text = processor.batch_decode(completions, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return text

if __name__ == "__main__":
    run_all_tasks(eval_batch_llama, DATA_DIR, OUT_DIR) 
