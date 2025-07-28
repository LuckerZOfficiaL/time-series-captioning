from functools import lru_cache
import json
import os
from pathlib import Path
import re
import time


from llava.eval.run_llava import image_parser, load_images 

from .inference_utils import run_all_tasks
from ..constants import DATA_ROOT

import requests
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

MODEL_PATH = "llava-hf/llava-v1.6-mistral-7b-hf"
FINETUNED_MODEL_PATH = "/shared/tsqa/finetuned_models/llava_lora_finetune/"
DATA_DIR = f"{DATA_ROOT}/data/samples/new samples no overlap/hard_questions_small/"
OUT_DIR = f"{DATA_ROOT}/finetuned_llava_inference_results_small/"


@lru_cache
def _load_batch_llava_model(model_name, device):
    from peft import AutoPeftModelForCausalLM
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        _attn_implementation='eager'
    )
    model.to(device)
    processor = LlavaNextProcessor.from_pretrained(MODEL_PATH, use_fast=True)
    return model, processor

def eval_batch_llava(prompts, image_files, device, use_image=True):
    model, processor = _load_batch_llava_model(FINETUNED_MODEL_PATH, device)
    print(f"use_image={use_image}")
    for i, p in enumerate(prompts):
        if "<image" in p:
            prompts[i] = re.sub(r"<image_(\d+)>", r"<|image_\1|>", p) 

    if use_image:
        images = [[Image.open(fn) for fn in images] for images in image_files]
        conversations = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": img} for img in curr_images 
                ] + [
                    {"type": "text", "text": f"{prompt}"},
                ],
        } for prompt, curr_images in zip(prompts, images)]
        prompts = [processor.apply_chat_template([c], add_generation_prompt=True)
                   for c in conversations]
        inputs = processor(images=images, text=prompts, padding=True, return_tensors="pt").to(device)
    else:
        conversations = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{prompt}"},
                ],
        } for prompt in prompts]
        prompts = [processor.apply_chat_template([c], add_generation_prompt=True)
                   for c in conversations]
        inputs = processor(text=prompts, padding=True, return_tensors="pt").to(device)

    stime = time.time()
    with torch.no_grad():
        generate_ids = model.generate(**inputs, max_new_tokens=20, temperature=0.3, do_sample=True)
    print(f"RUNTIME on {device}: {time.time() - stime:.2f} seconds")    
    results = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    # Remove original prompt from returned result
    captions = [r.split('[/INST] ')[1] for r in results]
    return captions

if __name__ == "__main__":
    run_all_tasks(eval_batch_llava, DATA_DIR, OUT_DIR)
