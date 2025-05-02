from functools import lru_cache
import json
import os
from pathlib import Path
import re
import time

import requests
from PIL import Image
import torch

from lmdeploy import pipeline, TurbomindEngineConfig
from lmdeploy.vl import load_image
from transformers import AutoModelForCausalLM, AutoTokenizer

from helpers import generate_prompt_for_baseline
from phi_parallel_gpu import main 

MODEL_PATH = "OpenGVLab/InternVL2-2B"
DATA_DIR = "/home/ubuntu/time-series-captioning/data/samples/len 10"
OUT_DIR = "/home/ubuntu/time-series-captioning/internvl_etiology"


@lru_cache
def _load_batch_internVL_model(model_name, device):
    torch.manual_seed(314)
    pipe = pipeline(model_name, backend_config=TurbomindEngineConfig(session_len=8192))   # orig 8192 
    return pipe 

def eval_batch_internVL(prompts, image_files, device, use_image): 
    print(f"use_image={use_image}")
    pipe = _load_batch_internVL_model(MODEL_PATH, device)
    if use_image:
        images = [Image.open(fn) for fn in image_files]
        prompts = [(prompt, image) for prompt, image in zip(prompts, images)]
    
    # Batch Inference
    stime = time.time()
    results = pipe(prompts, max_new_tokens=20, temperature=0.3)
    print(f"RUNTIME on {device}: {time.time() - stime:.2f} seconds")
    return [r.text for r in results] 

if __name__ == "__main__":
    main(eval_batch_internVL, DATA_DIR, OUT_DIR, use_image=True)
