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
from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info 

from .inference_utils import run_all_tasks, run_PAL_captions

#MODEL_PATH = "Qwen/Qwen2.5-VL-32B-Instruct"
MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"
#FINETUNED_PATH = "/shared/tsqa/finetuned_models/qwenVL_fine_tune"
#DATA_DIR = "/home/ubuntu/cats-bench/time-series-captioning/easy_subsample" 
DATA_DIR = "/shared/tsqa/CaTSBench/all_questions"
OUT_DIR = "/home/ubuntu/time-series-captioning/qwenvl_inference_results_all"
#DATA_DIR="caption_prompts/"
#OUT_DIR = "qwen32B_captions"

@lru_cache
def _load_batch_qwenVL_model(model_name, device):
    torch.manual_seed(31444)
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,                      # or load_in_8bit=True
        bnb_4bit_quant_type="nf4",              # best accuracy / size trade-off
        bnb_4bit_compute_dtype=torch.bfloat16,  # math keeps BF16 precision
        bnb_4bit_use_double_quant=True,         # second quant stage = better quality
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                #"qwen2.5-vl-4bit-fast",  # quantized version for fast loading
                MODEL_PATH
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    processor.tokenizer.padding_side = 'left'  # Fix padding for decoder-only architecture
    return model, processor


def eval_batch_qwenVL(prompts, image_files, device, use_image): 
    print(f"use_image={use_image}")
    for i, p in enumerate(prompts):
        if "<image" in p:
            prompts[i] = re.sub(r"<image_(\d+)>", r"<|image_\1|>", p)

    model, processor = _load_batch_qwenVL_model(MODEL_PATH, device)
    if use_image:
        content_list = [
            [{"type": "image", "image": image_file} for image_file in image_batch] + \
            [{"type": "text", "text": prompt}] for prompt, image_batch in zip(prompts, image_files)
        ]
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
        images = [Image.open(fn) for curr_images in image_files for fn in curr_images]
        image_inputs, _ = process_vision_info(conversations)
    else:
        image_inputs = None
    inputs = processor(text=prompts, images=image_inputs, padding=True, return_tensors="pt") 
    inputs = inputs.to(model.device).to(model.dtype)

    # Batch Inference
    stime = time.time()
    with torch.no_grad():
        text_ids = model.generate(**inputs, max_new_tokens=3072, temperature=0.3, do_sample=True)
    prompt_len = inputs["input_ids"].shape[1]
    completions = text_ids[:, prompt_len:]
    print(f"RUNTIME on {device}: {time.time() - stime:.2f} seconds")
    text = processor.batch_decode(completions, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return text

if __name__ == "__main__":
    run_all_tasks(eval_batch_qwenVL, DATA_DIR, OUT_DIR)
    #run_PAL_captions(eval_batch_qwenVL, "test_caption_prompts.json", 
    #                plots_dir="data/samples/new samples no overlap/test/plots", 
    #                out_dir="qwen_PAL_captions")
 
