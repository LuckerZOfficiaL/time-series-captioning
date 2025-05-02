from functools import lru_cache
from itertools import count
import json
import os
from pathlib import Path
import re
import time

import requests
from PIL import Image
from qwen_omni_utils import process_mm_info
import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from transformers import AutoModelForCausalLM, AutoTokenizer

from source.multi_gpu_utils import run_multi_gpu

MODEL_PATH = "Qwen/Qwen2.5-Omni-7B"
DATA_DIR = "/home/ubuntu/time-series-captioning/data/samples/new samples no overlap/tasks"
OUT_DIR = "/home/ubuntu/time-series-captioning/qwen_inference_results"

@lru_cache
def _load_batch_qwen_model(model_name, device):
    torch.manual_seed(314)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16,
                                                                _attn_implementation="sdpa",
                                                                low_cpu_mem_usage=True)
    model.to(device)
    processor = Qwen2_5OmniProcessor.from_pretrained(model_name)
    return model, processor

def renumber_images(text: str) -> str:
    """
    Replace each literal '<image>' in the input text with
    '<|image_1|>', '<|image_2|>', … in sequential order.
    """
    counter = count(1)  # will yield 1, 2, 3, …
    def _replace(match):
        i = next(counter)
        return f"<|image_{i}|>"
    return re.sub(r"<image>", _replace, text)


def eval_batch_qwen(prompts, image_files, device, use_image): 
    print(f"use_image={use_image}")
    for i, p in enumerate(prompts):
        if "<image>" in p:
            prompts[i] = renumber_images(p)
            
    model, processor = _load_batch_qwen_model(MODEL_PATH, device)
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
                "role": "system",
                "content": [{"type": "text", "text": ("You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group,"
                             " capable of perceiving auditory and visual inputs, as well as generating text and speech.")}]
            },
            {
                "role": "user",
                "content": content,
            }] for content in content_list 
    ]
    print(conversations[0])
    text = processor.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)
    print(text)
    if use_image:
        _, images, _ = process_mm_info(conversations, use_audio_in_video=False)
        inputs = processor(text=text, images=images, return_tensors="pt", padding=True, use_audio_in_video=False)
    else:
        inputs = processor(text=text, return_tensors="pt", padding=True, use_audio_in_video=False)
    inputs = inputs.to(model.device).to(model.dtype)

    # Batch Inference
    stime = time.time()
    text_ids = model.generate(**inputs, max_new_tokens=20, temperature=0.3, do_sample=True, 
                              use_audio_in_video=False, return_audio=False)
    print(f"RUNTIME on {device}: {time.time() - stime:.2f} seconds")
    text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    captions = [t.split("assistant\n")[1] for t in text]
    return captions

if __name__ == "__main__":
    use_image = True
    task = "caption_retrieval_cross_domain"
    out_dir_name = task + ("_no_image" if not use_image else "_with_image") 
    run_multi_gpu(eval_batch_qwen, os.path.join(DATA_DIR, task), os.path.join(OUT_DIR, out_dir_name), use_image=use_image)
