from functools import lru_cache
import json
import os
from pathlib import Path
import re
import time


from llava.eval.run_llava import image_parser, load_images 

from helpers import generate_prompt_for_baseline

MODEL_PATH = "llava-hf/llava-v1.6-mistral-7b-hf"
DATA_DIR = "/home/ubuntu/time-series-captioning/data/samples/"
OUT_DIR = "/home/ubuntu/time-series-captioning/llava_captions_text"


import requests
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig


@lru_cache
def _load_batch_llava_model(model_name):
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to('cuda')
    processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
    return model, processor

def eval_batch_llava(prompts, image_files):
    conversations = [{
            "role": "user",
            "content": [
#                {"type": "image"},
                {"type": "text", "text": f"{prompt}"},
            ],
    } for prompt in prompts]
    model, processor = _load_batch_llava_model(MODEL_PATH)
    prompts = [processor.apply_chat_template([c], add_generation_prompt=True)
               for c in conversations]
    inputs = processor(text=prompts, padding=True, return_tensors="pt").to(model.device)
    stime = time.time()
    generate_ids = model.generate(**inputs, max_new_tokens=256, temperature=0.3, do_sample=True)
    print("BATCH TIME:", time.time() - stime)
    results = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    # Remove original prompt from returned result
    captions = [r.split('[/INST] ')[1] for r in results]
    return captions


def write_caption(ts_names, eval_fn):
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

    captions = eval_fn(prompts, image_files)
    for ts_name, caption in zip(ts_names, captions):
        with open(os.path.join(OUT_DIR, f"{ts_name}.txt"), "w+") as fh:
            fh.write(caption)


def main(eval_fn):
    ts_names = [Path(fn).stem for fn in os.listdir(os.path.join(DATA_DIR, "time series"))]
    done_names = {Path(fn).stem for fn in os.listdir(OUT_DIR)}
    ts_names = sorted([name for name in ts_names if name not in done_names])
    batch_size = 10
    for i in range(0, len(ts_names), batch_size):
        ts_batch = ts_names[i:i+batch_size]
        write_caption(ts_batch, eval_fn)

if __name__ == "__main__":
    main(eval_batch_llava)
