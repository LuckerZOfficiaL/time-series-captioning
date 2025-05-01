from collections import defaultdict
import json
import pickle
import os

import numpy as np

def load_data(data_path):
    data = []
    caption_paths = sorted(os.listdir(os.path.join(data_path, 'gt_captions')))
    ts_paths = sorted(os.listdir(os.path.join(data_path, 'time series')))
    assert caption_paths == ts_paths
    for path in ts_paths:
        ts_name = path.split('.')[0]
        with open(os.path.join(data_path, 'time series', path)) as fh:
            ts_strings = fh.read().split('\n')
            ts_strings.remove('')
            ts = [float(x) for x in ts_strings]
        with open(os.path.join(data_path, 'gt_captions', path)) as fh:
            caption = fh.read()
        data.append({
            "ts_name": ts_name,
            "ts": ts,
            "caption": caption,
            "plot": os.path.join(data_path, 'plots', path.replace('.txt', '.jpeg'))
        })
    return data


def group_data(data, groups):
    grouped_data = defaultdict(list)
    for ts_data in data: 
        [dataset] = [d for d in groups if d in ts_data["ts_name"]]
        grouped_data[dataset].append(ts_data)
    return grouped_data


def run_prompt_creator(make_prompts, data_path, out_dir, groups=None):
    data = load_data(data_path)
    if groups:
        grouped_data = group_data(data, groups)
        all_prompts_nested = [make_prompts(data_group) for data_group in grouped_data.values()]
        all_prompts = [item for sublist in all_prompts_nested for item in sublist]
    else:
        all_prompts = make_prompts(data) 
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "tasks.json"), "w") as fh:
        json.dump(all_prompts, fh)

