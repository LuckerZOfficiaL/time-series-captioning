from functools import lru_cache
import json
import os
from pathlib import Path
import re
import time


from llava.eval.run_llava import image_parser, load_images 

from source.helpers import generate_prompt_for_baseline
from source.multi_gpu_utils import caption_loader, task_loader, run_multi_gpu

import requests
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

MODEL_PATH = "llava-hf/llava-v1.6-mistral-7b-hf"
DATA_DIR = "/home/ubuntu/time-series-captioning/data/samples/new samples no overlap/tasks/caption_retrieval_cross_domain_with_image"
# TODO: maybe name this out_dir automatically?
OUT_DIR = "/home/ubuntu/time-series-captioning/llava_caption_retrieval_with_image_easy"


@lru_cache
def _load_batch_llava_model(model_name, device):
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        _attn_implementation='eager'
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
    return model, processor

def eval_batch_llava(prompts, image_files, device, use_image=True):
    model, processor = _load_batch_llava_model(MODEL_PATH, device)
    print(f"use_image={use_image}")
    if use_image:
        conversations = [{
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"{prompt}"},
                ],
        } for prompt in prompts]
        images = [Image.open(fn[0]) for fn in image_files]
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
    generate_ids = model.generate(**inputs, max_new_tokens=20, temperature=0.3, do_sample=True)
    print(f"RUNTIME on {device}: {time.time() - stime:.2f} seconds")    
    results = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    # Remove original prompt from returned result
    captions = [r.split('[/INST] ')[1] for r in results]
    return captions

if __name__ == "__main__":
    run_multi_gpu(eval_batch_llava, task_loader, DATA_DIR, OUT_DIR, use_image=True)
