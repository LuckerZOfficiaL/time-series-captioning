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
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig


@lru_cache
def _load_batch_deepseek_model(model_name, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    model.to(device)
    tokenizer.pad_token = tokenizer.eos_token    
    model.generation_config = GenerationConfig.from_pretrained(model_name)
    model.generation_config.pad_token_id = model.generation_config.eos_token_id
    return model, tokenizer


def eval_batch_deepseek(prompts, image_files, device, use_image=True):
    model, tokenizer = _load_batch_deepseek_model(MODEL_PATH, device)
    messages = [{"role": "user", "content": prompt} for prompt in prompts]
    input_tensor = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")

    stime = time.time()
    generate_ids = model.generate(input_tensor.to(model.device), max_new_tokens=1024) 
#                                  temperature=0.3, do_sample=True)
    print(f"RUNTIME on {device}: {time.time() - stime:.2f} seconds")    
    captions = [tokenizer.decode(output[input_tensor.shape[1]:], skip_special_tokens=True) for output in generate_ids]
    captions = [c.replace(p, "") for c, p in zip(captions, prompts)]
    print(captions[0])
    return captions


if __name__ == "__main__":
    main(eval_batch_deepseek, DATA_DIR, OUT_DIR, use_image=False)
