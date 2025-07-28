from functools import lru_cache
import json
import os
from pathlib import Path
import re
import time

import requests
from PIL import Image
import torch
from transformers import AutoTokenizer, AutoModel
from transformers import pipeline

from lmdeploy import pipeline, TurbomindEngineConfig
from lmdeploy.vl import load_image
from lmdeploy.vl.constants import IMAGE_TOKEN

from .inference_utils import run_all_tasks

MODEL_PATH = "OpenGVLab/InternVL2_5-8B"
#MODEL_PATH = "/shared/tsqa/finetuned_models/internvl_8b_finetune"
DATA_DIR = "/home/ubuntu/cats-bench/time-series-captioning/easy_subsample" 
OUT_DIR = "/home/ubuntu/time-series-captioning/internvl_inference_easy" 

@lru_cache
def _load_batch_internVL_model(model_name, device):
    torch.manual_seed(314)
    pipe = pipeline(model_name, backend_config=TurbomindEngineConfig(session_len=8192))   # orig 8192 
#    pipe = pipeline(
#    "vqa",
#    model=MODEL_PATH,
#    tokenizer=MODEL_PATH,
#    trust_remote_code=True,
#    device=device,
#    )
    return pipe 

def eval_batch_internVL(prompts, image_files, device, use_image): 
    print(f"use_image={use_image}")
    pipe = _load_batch_internVL_model(MODEL_PATH, device)
    for i, p in enumerate(prompts):
        if "<image" in p:
            prompts[i] = re.sub(r"<image_(\d+)>", IMAGE_TOKEN, p)    
    if use_image:
        images = [[Image.open(fn) for fn in curr_images] for curr_images in image_files]
        prompts = [(prompt, curr_images) for prompt, curr_images in zip(prompts, images)]
    
    # Batch Inference
    stime = time.time()
    results = pipe(prompts, max_new_tokens=20, temperature=0.3)
    print(f"RUNTIME on {device}: {time.time() - stime:.2f} seconds")
    return [r.text for r in results] 

if __name__ == "__main__":
    run_all_tasks(eval_batch_internVL, DATA_DIR, OUT_DIR)
