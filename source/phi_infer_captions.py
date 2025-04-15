from functools import lru_cache
import json
import os
from pathlib import Path
import re
import time



from helpers import generate_prompt_for_baseline

MODEL_PATH = "microsoft/Phi-4-multimodal-instruct"
DATA_DIR = "/home/ubuntu/time-series-captioning/data/samples/new samples no overlap/test"
OUT_DIR = "/home/ubuntu/time-series-captioning/phi_captions_test"


import requests
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

import requests
import torch
import os
import io
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from urllib.request import urlopen


@lru_cache
def _load_batch_phi_model(model_name):
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map="cuda", 
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        _attn_implementation='flash_attention_2',
    ).cuda()
    return model, processor

def eval_batch_phi(prompts, image_files):
    # Load generation config
    generation_config = GenerationConfig.from_pretrained(MODEL_PATH)
    model, processor = _load_batch_phi_model(MODEL_PATH)
    
    # Define prompt structure
    user_prompt = '<|user|>'
    assistant_prompt = '<|assistant|>'
    prompt_suffix = '<|end|>'
    
    images = [Image.open(fn) for fn in image_files]
    prompts = [f'{user_prompt}<|image_1|>{caption_prompt}{prompt_suffix}{assistant_prompt}'
               for caption_prompt in prompts]
    inputs = processor(text=prompts, images=images, return_tensors='pt').to('cuda')
    
    # Generate response
    stime = time.time()
    generate_ids = model.generate(
        **inputs,
        max_new_tokens=256,
        generation_config=generation_config,
        temperature=0.3,
        do_sample=True,
        num_logits_to_keep=0,
    )
    print("RUNTIME", time.time() - stime)
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    captions = processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return captions


def write_caption(ts_names):
    """
    Given ts_name, write a caption .txt file for the time series.
    """
    prompts = []
    image_files = []
    for ts_name in ts_names:
      dataset_name = ts_name.split("_")[0]
      with open(os.path.join(DATA_DIR, "metadata", f"{ts_name}.json"), 'r') as fh:
          metadata = json.load(fh)
      with open(os.path.join(DATA_DIR, "time series", f"{ts_name}.txt"), 'r') as fh:
          ts = fh.read()
      prompt = generate_prompt_for_baseline(dataset_name, metadata, ts, use_image=False)
      image_file = os.path.join(DATA_DIR, "plots_2.0", f"{ts_name}.jpeg")
      prompts.append(prompt)
      image_files.append(image_file)

    captions = eval_batch_phi(prompts, image_files)
    for ts_name, caption in zip(ts_names, captions):
        with open(os.path.join(OUT_DIR, f"{ts_name}.txt"), "w+") as fh:
            fh.write(caption)


def main():
    ts_names = [Path(fn).stem for fn in os.listdir(os.path.join(DATA_DIR, "time series"))]
    done_names = {Path(fn).stem for fn in os.listdir(OUT_DIR)}
    ts_names = sorted([name for name in ts_names if name not in done_names])
    batch_size = 10
    for i in range(0, len(ts_names), batch_size):
        ts_batch = ts_names[i:i+batch_size]
        write_caption(ts_batch)

if __name__ == "__main__":
    main()
