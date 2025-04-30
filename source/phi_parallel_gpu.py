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


def write_caption(model_eval, ts_names, device: torch.device, data_dir, out_dir, use_image=True):
    prompts = []
    image_files = []
    for ts_name in ts_names:
        dataset_name = ts_name.split("_")[0]
        metadata_path = os.path.join(data_dir, "metadata", f"{ts_name}.json")
        ts_file_path = os.path.join(data_dir, "time series", f"{ts_name}.txt")
        with open(metadata_path, 'r') as fh:
            metadata = json.load(fh)
        with open(ts_file_path, 'r') as fh:
            ts = fh.read()
        prompt = generate_prompt_for_baseline(dataset_name, metadata, ts, use_image=use_image)
        image_file = os.path.join(data_dir, "plots", f"{ts_name}.jpeg")
        prompts.append(prompt)
        image_files.append(image_file)

    captions = model_eval(prompts, image_files, device, use_image)
    print(f"WRITING TO {out_dir}")
    for ts_name, caption in zip(ts_names, captions):
        out_file = os.path.join(out_dir, f"{ts_name}.txt")
        with open(out_file, "w+") as fh:
            fh.write(caption)

# TODO: refactor this into existing code
def write_caption_etiology(model_eval, ts_batch, device: torch.device, data_dir, out_dir):
    use_image = True
    prompts = []
    images = []
    for ts_name in ts_batch:
        with open(os.path.join(data_dir, "prompts", f"{ts_name}.json")) as fh:
            inputs = json.load(fh)
            prompts.append(inputs['prompt']) 
            if use_image: 
                images.append(inputs['plot_paths'])       
    # TODO: parametrize use_image 
    captions = model_eval(prompts, image_files=images, device=device, use_image=use_image)
    for ts_name, caption in zip(ts_batch, captions):
        out_file = os.path.join(out_dir, f"{ts_name}.txt")
        with open(out_file, "w+") as fh:
            fh.write(caption)

def process_worker(gpu_id, model_eval, ts_names, data_dir, out_dir, use_image=True):
    device = torch.device(f"cuda:{gpu_id}")
    print(f"Process started on GPU {gpu_id} for {len(ts_names)} time series.")
    for i in range(0, len(ts_names), BATCH_SIZE):
        ts_batch = ts_names[i:i+BATCH_SIZE]
        write_caption_etiology(model_eval, ts_batch, device, data_dir, out_dir)
    print(f"Process on GPU {gpu_id} finished processing {len(ts_names)} time series.")


NUM_GPUS_TO_USE = 1

def main(model_eval, data_dir, out_dir, use_image=True):
    # Retrieve list of time series names yet to be processed.
    in_dir = "prompts"
    ts_dir = os.path.join(data_dir, in_dir)
    ts_names = [Path(fn).stem for fn in os.listdir(ts_dir)]
    done_names = {Path(fn).stem for fn in os.listdir(out_dir)}
    ts_names = sorted([name for name in ts_names if name not in done_names])

    # Select the top NUM_GPUS_TO_USE GPUs based on available (free) memory.
    gpus = GPUtil.getGPUs()
    if len(gpus) < NUM_GPUS_TO_USE:
        print(f"Warning: Less than {NUM_GPUS_TO_USE} GPUs detected; using all available GPUs.")
    gpus_sorted = sorted(gpus, key=lambda gpu: gpu.memoryFree, reverse=True)
    selected_gpu_ids = [gpu.id for gpu in gpus_sorted][:NUM_GPUS_TO_USE] 
    print(f"Selected GPUs (by available memory): {selected_gpu_ids}")

    # Partition tasks among the selected GPUs using a round-robin assignment.
    gpu_assignments = {gpu_id: [] for gpu_id in selected_gpu_ids}
    for idx, name in enumerate(ts_names):
        gpu_id = selected_gpu_ids[idx % len(selected_gpu_ids)]
        gpu_assignments[gpu_id].append(name)
    
    # Print the number of tasks per GPU.
    for gpu_id, assigned_tasks in gpu_assignments.items():
        print(f"GPU {gpu_id} assigned {len(assigned_tasks)} time series.")
    
    # Create a multiprocessing context that uses the spawn start method.
    spawn_ctx = mp.get_context("spawn")

    # Launch a worker process for each selected GPU.
    with ProcessPoolExecutor(max_workers=len(selected_gpu_ids), mp_context=spawn_ctx) as executor:
        futures = []
        for gpu_id, assigned_names in gpu_assignments.items():
            futures.append(executor.submit(process_worker, gpu_id, model_eval, assigned_names, data_dir, out_dir, use_image))
        for future in futures:
            future.result()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    USE_IMAGE = False
    main(eval_batch_phi, DATA_DIR, OUT_DIR, use_image=USE_IMAGE)

