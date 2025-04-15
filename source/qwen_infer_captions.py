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
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from transformers import AutoModelForCausalLM, AutoTokenizer

from helpers import generate_prompt_for_baseline
from infer_model_captions import main

MODEL_PATH = "Qwen/Qwen2.5-Omni-7B"
DATA_DIR = "/home/ubuntu/time-series-captioning/data/samples/"
OUT_DIR = "/home/ubuntu/time-series-captioning/qwen_captions"


@lru_cache
def _load_batch_qwen_model(model_name):
    torch.manual_seed(314)
#    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
#        model_name,
#        torch_dtype="auto",
#        device_map="auto",
#        attn_implementation="flash_attention_2",
#    ).to('cuda')
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-Omni-7B", torch_dtype="auto").to('cuda') 
    processor = Qwen2_5OmniProcessor.from_pretrained(model_name)
    return model, processor

def eval_batch_qwen(prompts, image_files): 
    model, processor = _load_batch_qwen_model(MODEL_PATH)
    captions = []
    for prompt, image_file in zip(prompts, image_files):
        conversation = [
            {
                "role": "system",
                "content": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_file},
            {"type": "text", "text": prompt}
                ],
            },
        ]
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        _, images, _ = process_mm_info(conversation, use_audio_in_video=False)
        inputs = processor(text=text, images=images, return_tensors="pt", padding=True).to(model.device)
        text_ids, _ = model.generate(**inputs, max_new_tokens=256, temperature=0.3, do_sample=True)
        text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        import pdb; pdb.set_trace()

    import pdb; pdb.set_trace()
    

    stime = time.time()
    print("BATCH TIME:", time.time() - stime)
    results = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    # Remove original prompt from returned result
    captions = [r.split('[/INST] ')[1] for r in results]
    return captions

if __name__ == "__main__":
    main(eval_batch_qwen)
