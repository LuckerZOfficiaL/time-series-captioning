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

from source.helpers import generate_prompt_for_baseline

# Adjust these as needed for memory constraints
BATCH_SIZE = 10
NUM_GPUS_TO_USE = 2


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


def write_caption(model_eval, tasks, device, data_dir, out_dir, use_image=True):
    def _get_prompt(task, use_image):
        prompt_name = "prompt_with_image" if use_image else "prompt_no_image"
        if prompt_name not in task:
            if use_image:
                raise ValueError
            prompt_name = "prompt"
        return task[prompt_name] 
    prompts = [_get_prompt(t, use_image) for t in tasks]
    image_files = [t["image_paths"] for t in tasks] if use_image else []

    captions = model_eval(prompts, image_files, device, use_image)
    for ts, caption in zip(tasks, captions):
        ts_name = ts["task_id"]
        out_file = os.path.join(out_dir, f"{ts_name}.txt")
        with open(out_file, "w+") as fh:
            fh.write(caption)

def process_worker(gpu_id, model_eval, tasks, data_dir, out_dir, use_image=True, handler_fn=None):
    if handler_fn is None:
        handler_fn = write_caption
    device = torch.device(f"cuda:{gpu_id}")
    print(f"Process started on GPU {gpu_id} for {len(tasks)} time series.")
    for i in range(0, len(tasks), BATCH_SIZE):
        ts_batch = tasks[i:i+BATCH_SIZE]
        handler_fn(model_eval, ts_batch, device, data_dir, out_dir, use_image)
    print(f"Process on GPU {gpu_id} finished processing {len(tasks)} time series.")


def run_multi_gpu(model_eval, data_dir, out_dir, use_image=True, handler_fn=None):
    # Retrieve list of time series names yet to be processed.
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(data_dir, "tasks.json")) as fh:
        tasks = json.load(fh)

    done_names = {Path(fn).stem for fn in os.listdir(out_dir)}
    remaining_ts = sorted([t for t in tasks if t["ts_name"] not in done_names], 
                          key=lambda t: t["ts_name"])
    if not remaining_ts:
        print("Task complete, returning")
        return
    # Select the top NUM_GPUS_TO_USE GPUs based on available (free) memory.
    gpus = GPUtil.getGPUs()
    if len(gpus) < NUM_GPUS_TO_USE:
        print(f"Warning: Less than {NUM_GPUS_TO_USE} GPUs detected; using all available GPUs.")
    gpus_sorted = sorted(gpus, key=lambda gpu: gpu.memoryFree, reverse=True)
    selected_gpu_ids = [gpu.id for gpu in gpus_sorted][:NUM_GPUS_TO_USE] 
    print(f"Selected GPUs (by available memory): {selected_gpu_ids}")

    # Partition tasks among the selected GPUs using a round-robin assignment.
    gpu_assignments = {gpu_id: [] for gpu_id in selected_gpu_ids}
    for idx, task in enumerate(remaining_ts):
        gpu_id = selected_gpu_ids[idx % len(selected_gpu_ids)]
        gpu_assignments[gpu_id].append(task)
    
    # Print the number of tasks per GPU.
    for gpu_id, assigned_tasks in gpu_assignments.items():
        print(f"GPU {gpu_id} assigned {len(assigned_tasks)} time series.")
    
    # Create a multiprocessing context that uses the spawn start method.
    spawn_ctx = mp.get_context("spawn")
 
    # Launch a worker process for each selected GPU.
    with ProcessPoolExecutor(max_workers=len(selected_gpu_ids), mp_context=spawn_ctx) as executor:
        futures = []
        for gpu_id, assigned_tasks in gpu_assignments.items():
            # Can use for single-process debugging:
            #process_worker(gpu_id, model_eval, assigned_tasks, data_dir, out_dir, use_image, handler_fn)
            futures.append(executor.submit(process_worker, gpu_id, model_eval, assigned_tasks, data_dir, out_dir, use_image, handler_fn))
        for future in futures:
            future.result()

