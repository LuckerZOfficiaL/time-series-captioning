from functools import lru_cache
import json
import os
from pathlib import Path
import re
import time



from helpers import generate_prompt_for_baseline
from phi_parallel_gpu import main

MODEL_PATH = "deepseek-ai/deepseek-math-7b-instruct"
DATA_DIR = "/home/ubuntu/time-series-captioning/data/samples/new samples no overlap/test"
OUT_DIR = "/home/ubuntu/time-series-captioning/deepseek-math-7b_text"


import requests
from PIL import Image
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, pipeline


@lru_cache
def _load_batch_deepseek_model(model_name, device):
    torch.manual_seed(314)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    model.to(device)
    tokenizer.pad_token = tokenizer.eos_token    
    model.generation_config = GenerationConfig.from_pretrained(model_name)
    model.generation_config.pad_token_id = model.generation_config.eos_token_id
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
        torch_dtype=torch.bfloat16,
        return_full_text=False,
        batch_size=10,           # default batch size
    )    
    return model, tokenizer, generator


def eval_batch_deepseek(prompts, image_files, device, use_image=True):
    _, _, generator = _load_batch_deepseek_model(MODEL_PATH, device)
    stime = time.time()
    with torch.inference_mode():
        results = generator(prompts, max_new_tokens=256, temperature=0.3, do_sample=True)
    print(f"RUNTIME on {device}: {time.time() - stime:.2f} seconds")
    captions = [r[0]["generated_text"] for r in results]
    print(captions[0])
    return captions


if __name__ == "__main__":
    main(eval_batch_deepseek, DATA_DIR, OUT_DIR, use_image=False)
