from functools import lru_cache
import json
import os
from pathlib import Path
import re
import time


from llava.eval.run_llava import image_parser, load_images 

from helpers import generate_prompt_for_baseline
from phi_parallel_gpu import main

MODEL_PATH = "llava-hf/llava-v1.6-mistral-7b-hf"
DATA_DIR = "/home/ubuntu/time-series-captioning/data/samples/new samples no overlap/test"
OUT_DIR = "/home/ubuntu/time-series-captioning/llava_captions_test"


import requests
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig


@lru_cache
def _load_batch_llava_model(model_name, device):
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
    return model, processor

def eval_batch_llava(prompts, image_files, device, use_image=True):
    # TODO: Add logic for use_image=False
    print(f"use_image={use_image}")
    conversations = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": f"{prompt}"},
            ],
    } for prompt in prompts]
    model, processor = _load_batch_llava_model(MODEL_PATH, device)
    images = [Image.open(fn) for fn in image_files]
    prompts = [processor.apply_chat_template([c], add_generation_prompt=True)
               for c in conversations]
    inputs = processor(images=images, text=prompts, padding=True, return_tensors="pt").to(device)
    stime = time.time()
    generate_ids = model.generate(**inputs, max_new_tokens=256, temperature=0.3, do_sample=True)
    print(f"RUNTIME on {device}: {time.time() - stime:.2f} seconds")    
    results = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    # Remove original prompt from returned result
    captions = [r.split('[/INST] ')[1] for r in results]
    return captions


if __name__ == "__main__":
    main(eval_batch_llava, DATA_DIR, OUT_DIR, use_image=True)
