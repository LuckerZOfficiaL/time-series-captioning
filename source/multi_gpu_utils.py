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
import torch
from PIL import Image

from helpers import generate_prompt_for_baseline

# Adjust these as needed for memory constraints
BATCH_SIZE = 1 
NUM_GPUS_TO_USE = 1


# TODO: Save these caption baseline prompts to disk, then just load them directly
def caption_loader(ts_names, data_dir):
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
    return prompts, image_files

def task_loader(ts_names, data_dir):
    prompts = []
    image_files = []
    for ts_name in ts_names:
        with open(os.path.join(data_dir, "prompts", f"{ts_name}.json")) as fh:
            inputs = json.load(fh)
            prompts.append(inputs['prompt']) 
            image_files.append(inputs['plot_paths'])
    return prompts, image_files 


def write_caption(model_eval, ts_names, device, data_dir, out_dir, loader, use_image=True):
    prompts, image_files = loader(ts_names, data_dir)
    captions = model_eval(prompts, image_files, device, use_image)
    print(f"WRITING TO {out_dir}")
    for ts_name, caption in zip(ts_names, captions):
        out_file = os.path.join(out_dir, f"{ts_name}.txt")
        with open(out_file, "w+") as fh:
            fh.write(caption)

def process_worker(gpu_id, model_eval, loader, ts_names, data_dir, out_dir, use_image=True):
    device = torch.device(f"cuda:{gpu_id}")
    print(f"Process started on GPU {gpu_id} for {len(ts_names)} time series.")
    for i in range(0, len(ts_names), BATCH_SIZE):
        ts_batch = ts_names[i:i+BATCH_SIZE]
        write_caption(model_eval, ts_batch, device, data_dir, out_dir, loader)
    print(f"Process on GPU {gpu_id} finished processing {len(ts_names)} time series.")


def run_multi_gpu(model_eval, loader, data_dir, out_dir, use_image=True):
    # Retrieve list of time series names yet to be processed.
    os.makedirs(out_dir, exist_ok=True)
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
            futures.append(executor.submit(process_worker, gpu_id, model_eval, loader, assigned_names, data_dir, out_dir, use_image))
        for future in futures:
            future.result()

