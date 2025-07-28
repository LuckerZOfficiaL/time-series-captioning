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
from transformers import AutoProcessor, AutoModelForVision2Seq

from .inference_utils import run_all_tasks

MODEL_PATH = "HuggingFaceTB/SmolVLM-Instruct" 
FINETUNED_PATH = "/shared/tsqa/finetuned_models/smolvlm"
DATA_DIR = "data/samples/new samples no overlap/hard_questions_small"
OUT_DIR = "/home/ubuntu/time-series-captioning/finetune_smolvlm_inference_results_small"


@lru_cache
def _load_batch_smolVLM_model(model_name, device):
    torch.manual_seed(314)
    model = AutoModelForVision2Seq.from_pretrained(
                FINETUNED_PATH, 
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    return model, processor


def eval_batch_smolVLM(prompts, image_files, device, use_image): 
    print(f"use_image={use_image}")
    for i, p in enumerate(prompts):
        if "<image" in p:
            prompts[i] = re.sub(r"<image_(\d+)>", r"<|image_\1|>", p)

    model, processor = _load_batch_smolVLM_model(MODEL_PATH, device)
    if use_image:
        content_list = [
            [{"type": "image", "image": image_file} for image_file in image_batch] + \
            [{"type": "text", "text": prompt}] for prompt, image_batch in zip(prompts, image_files)]
    else:
        content_list = [[
            {"type": "text", "text": prompt}
        ] for prompt in prompts]

    conversations = [
       [{
           "role": "user",
           "content": content,
       }] for content in content_list
    ]
    prompts = processor.apply_chat_template(conversations, add_generation_prompt=True)
    if use_image:
        images = [[Image.open(fn) for fn in curr_images] for curr_images in image_files]
        inputs = processor(text=prompts, images=images, return_tensors="pt")
    else:
        inputs = processor(text=prompts, return_tensors="pt")
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
    run_all_tasks(eval_batch_smolVLM, DATA_DIR, OUT_DIR) 
