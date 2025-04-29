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
from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor
from transformers import AutoModelForCausalLM, AutoTokenizer

from helpers import generate_prompt_for_baseline
from phi_parallel_gpu import main 

MODEL_PATH = "Qwen/Qwen2.5-Omni-7B"
DATA_DIR = "/home/ubuntu/time-series-captioning/data/samples/new samples no overlap/test"
OUT_DIR = "/home/ubuntu/time-series-captioning/qwen_fine_tune_etiology_test"


@lru_cache
def _load_batch_qwen_model(model_name, device):
    torch.manual_seed(314)
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
                "qwen_fine_tune_2", 
                torch_dtype=torch.float16,
                _attn_implementation='flash_attention_2',
                low_cpu_mem_usage=True
    )
    model.to(device)
    processor = Qwen2_5OmniProcessor.from_pretrained(model_name)
    return model, processor

def eval_batch_qwen(prompts, image_files, device, use_image): 
    print(f"use_image={use_image}")
    model, processor = _load_batch_qwen_model(MODEL_PATH, device)
    if use_image:
        content_list = [[
            {"type": "image", "image": image_file},
            {"type": "text", "text": prompt}
        ] for prompt, image_file in zip(prompts, image_files)]
    else:
        content_list = [[
            {"type": "text", "text": prompt}
        ] for prompt in prompts]
 
    conversations = [
            [{
                "role": "system",
                "content": [{"type": "text", "text": ("You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group,"
                             " capable of perceiving auditory and visual inputs, as well as generating text and speech.")}]
            },
            {
                "role": "user",
                "content": content,
            }] for content in content_list 
    ]
    text = processor.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)
    if use_image:
        _, images, _ = process_mm_info(conversations, use_audio_in_video=False)
        inputs = processor(text=text, images=images, return_tensors="pt", padding=True)
    else:
        inputs = processor(text=text, return_tensors="pt", padding=True, use_audio_in_video=False)
    inputs = inputs.to(model.device).to(model.dtype)

    # Batch Inference
    stime = time.time()
    text_ids = model.generate(**inputs, max_new_tokens=10, temperature=0.3, do_sample=True)
    print(f"RUNTIME on {device}: {time.time() - stime:.2f} seconds")
    text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    captions = [t.split("assistant\n")[1] for t in text]
    return captions

if __name__ == "__main__":
    main(eval_batch_qwen, DATA_DIR, OUT_DIR, use_image=True)
