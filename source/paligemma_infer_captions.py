from functools import lru_cache
import json
import os
from pathlib import Path
import re
import time

import requests
from PIL import Image
import torch
from transformers import (
    PaliGemmaProcessor,
    PaliGemmaForConditionalGeneration,
)
from transformers.image_utils import load_image
from transformers import AutoModelForCausalLM, AutoTokenizer

from helpers import generate_prompt_for_baseline
from phi_parallel_gpu import main 

MODEL_PATH = "google/paligemma2-10b-pt-448"
DATA_DIR = "/home/ubuntu/time-series-captioning/data/samples/new samples no overlap/test"
OUT_DIR = "/home/ubuntu/time-series-captioning/paligemma_captions_test"


@lru_cache
def _load_batch_paligemma_model(model_name, device):
    torch.manual_seed(314)
    model = PaliGemmaForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.bfloat16) 
    processor = PaliGemmaProcessor.from_pretrained(model_name)    
    model.to(device)
    return model, processor

def eval_batch_paligemma(prompts, image_files, device, use_image): 
    print(f"use_image={use_image}")
    model, processor = _load_batch_paligemma_model(MODEL_PATH, device)
    images = [Image.open(fn) for fn in image_files]
    prompts = ["<image>" + prompt for prompt in prompts]
    model_inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True).to(torch.bfloat16).to(model.device)
    input_len = model_inputs["input_ids"].shape[-1]
    stime = time.time()
    generation = model.generate(**model_inputs, max_new_tokens=512, temperature=0.3, do_sample=True) 
    print(f"RUNTIME on {device}: {time.time() - stime:.2f} seconds")
    generation = generation[:, input_len:]
    captions = processor.batch_decode(generation, skip_special_tokens=True)
    print(prompts[0])
    print('-----------')
    print(captions[0])
    exit()
    return captions

if __name__ == "__main__":
    main(eval_batch_paligemma, DATA_DIR, OUT_DIR, use_image=True)
