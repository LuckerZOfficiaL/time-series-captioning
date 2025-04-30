"""
Inference code using chatgpt-o3-mini-high suggestions for GPU parallelization.
"""
from functools import lru_cache
import json
import os
from pathlib import Path
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import GPUtil  # Make sure to install this package: pip install gputil

from helpers import generate_prompt_for_baseline

MODEL_PATH = "microsoft/Phi-4-multimodal-instruct"
DATA_DIR = "data/samples/new samples no overlap/tasks/caption_retrieval_hard_with_image"
OUT_DIR = "/home/ubuntu/time-series-captioning/phi_etiology_test_with_image_hard"
BATCH_SIZE = 1 # Adjust batch size as needed

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM, GenerationConfig


@lru_cache(maxsize=None)
def _load_batch_phi_model(model_name, device: torch.device):
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        _attn_implementation="eager"
#        _attn_implementation='flash_attention_2',
    )
    model.to(device)
    return model, processor


def eval_batch_phi(prompts: list[str], image_files: list[str], device: torch.device, use_image: bool):
    generation_config = GenerationConfig.from_pretrained(MODEL_PATH)
    model, processor = _load_batch_phi_model(MODEL_PATH, device)

    # Define prompt structure
    user_prompt = '<|user|>'
    assistant_prompt = '<|assistant|>'
    prompt_suffix = '<|end|>'

    # Format each prompt using the input caption prompt from your helper
    if use_image:
        print("Using images as well as text inputs.")
        images = [Image.open(fn) for fn in image_files]
        formatted_prompts = [
            f'{user_prompt}<|image_1|>{caption_prompt}{prompt_suffix}{assistant_prompt}'
            for caption_prompt in prompts
        ]
        inputs = processor(text=formatted_prompts, images=images, return_tensors='pt').to(device)
    else:
        print("Using only text prompts, no images.")
        formatted_prompts = [
            f'{user_prompt}{caption_prompt}{prompt_suffix}{assistant_prompt}'
            for caption_prompt in prompts
        ]
        inputs = processor(text=formatted_prompts, return_tensors='pt').to(device)


    stime = time.time()
    generate_ids = model.generate(
        **inputs,
        max_new_tokens=20,
        generation_config=generation_config,
        temperature=0.3,
        do_sample=True,
        num_logits_to_keep=0,
    )
    print(f"RUNTIME on {device}: {time.time() - stime:.2f} seconds")
    # Remove prompt tokens from generated output
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    captions = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return captions


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    USE_IMAGE = False
    main(eval_batch_phi, DATA_DIR, OUT_DIR, use_image=USE_IMAGE)

