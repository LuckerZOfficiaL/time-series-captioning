from collections import defaultdict
import json
import pickle
import os

import numpy as np

DATA_PATH = "/home/ubuntu/reasoning_data/TS_Dataset.jsonl"

PROMPT_TEMPLATE = """
Here is a time series caption:
{cap}

What time series is best described by this caption? 
(A) {ts1}
(B) {ts2}
(C) {ts3}
(D) {ts4}

{image_str}
You must respond only with valid JSON, and no extra text or markdown.
The JSON schema is:
{{
  "answer": <string>
}}
<string> must be an answer string containing only A, B, C, or D.
Ensure your output parses as JSON with exactly one top-level object containing the answer field.
"""
IMAGE_STR = "I have also attached a line plot image of each time series to support you:\n(A) <image_1>\n(B) <image_2>\n(C) <image_3>\n(D) <image_4>\n"

def load_data(path=DATA_PATH):
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            records.append(json.loads(line))
    return records

def main_local_data(data_path, out_dir, use_image=False):
    all_captions = []
    caption_paths = sorted(os.listdir(os.path.join(data_path, 'gt_captions')))
    ts_paths = sorted(os.listdir(os.path.join(data_path, 'time series')))
    assert caption_paths == ts_paths
    for path in caption_paths: 
        with open(os.path.join(data_path, 'gt_captions', path)) as fh:
            all_captions.append(fh.read())
    all_ts = []
    for path in ts_paths:
        with open(os.path.join(data_path, 'time series', path)) as fh:
            ts_strings = fh.read().split('\n')
            ts_strings.remove('')
            ts = [float(x) for x in ts_strings]
            all_ts.append(ts) 
    plot_paths = [os.path.join(data_path, 'plots', path.replace(".txt", ".jpeg")) for path in caption_paths]
    all_prompts = []
    ground_truths = []
    all_plots = []
    for i, cap in enumerate(all_captions):
        random_ts = np.random.randint(0, len(all_ts), size=3)
        while i in random_ts: 
            random_ts = np.random.randint(0, len(all_ts), size=3)
        time_series = [all_ts[i]] + [all_ts[z] for z in random_ts]
        np.random.shuffle(time_series)
        ground_truth = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}[time_series.index(all_ts[i])]
        ground_truths.append(ground_truth)
        ts_strings = [','.join([f"{x:.2f}" for x in ts]) for ts in time_series]
        prompt = PROMPT_TEMPLATE.format(cap=cap, image_str=(IMAGE_STR if use_image else ""), ts1=time_series[0],
                                        ts2=time_series[1], ts3=time_series[2], ts4=time_series[3])
        all_prompts.append(prompt)
        all_plots.append([plot_paths[i]] + [plot_paths[z] for z in random_ts])
    all_prompts = list(zip(all_prompts, all_plots)) 
    write_prompts(all_prompts, ground_truths, out_dir, ts_paths, use_image)


def write_prompts(all_prompts, ground_truths, out_dir, ts_paths, use_image):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "prompts"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "ground_truth"), exist_ok=True)
    for prompt, gt, name in zip(all_prompts, ground_truths, ts_paths):
        prompt_text, plot_path = prompt
        json_entry = {"prompt": prompt_text}
        if use_image:
            json_entry["plot_paths"] = plot_path
        with open(os.path.join(out_dir, "prompts", name.replace('.txt', '.json')), "w") as fh:
            json.dump(json_entry, fh) 
        with open(os.path.join(out_dir, "ground_truth", name), "w") as fh:
            fh.write(gt)  


def main_local_data_hard(data_path, out_dir, use_image=True):
    DATASETS = ["air quality", "crime", "border crossing", "demography", "road injuries", "covid",
                "co2", "diet", "online retail", "walmart", "agriculture"]
    all_captions = []
    caption_paths = sorted(os.listdir(os.path.join(data_path, 'gt_captions')))
    ts_paths = sorted(os.listdir(os.path.join(data_path, 'time series')))
    assert caption_paths == ts_paths
    all_captions = defaultdict(list)
    for path in caption_paths: 
        with open(os.path.join(data_path, 'gt_captions', path)) as fh:
            [dataset] = [d for d in DATASETS if d in path]
            all_captions[dataset].append(fh.read())
    all_ts = defaultdict(list)
    for path in ts_paths:
        with open(os.path.join(data_path, 'time series', path)) as fh:
            [dataset] = [d for d in DATASETS if d in path]
            ts_strings = fh.read().split('\n')
            ts_strings.remove('')
            ts = [float(x) for x in ts_strings]
            all_ts[dataset].append(ts) 
    orig_plot_paths = [os.path.join(data_path, 'plots', path.replace(".txt", ".jpeg")) for path in caption_paths]
    plot_paths = defaultdict(list)
    for path in orig_plot_paths:
        [dataset] = [d for d in DATASETS if d in path]
        plot_paths[dataset].append(path)    
    all_prompts = []
    ground_truths = []
    all_plots = []
    for dataset, curr_captions in all_captions.items():
        curr_ts = all_ts[dataset]
        for i, cap in enumerate(curr_captions):
            random_ts = np.random.randint(0, len(curr_ts), size=3)
            while i in random_ts: 
                random_ts = np.random.randint(0, len(curr_ts), size=3)
            time_series = [curr_ts[i]] + [curr_ts[z] for z in random_ts]
            np.random.shuffle(time_series)
            ground_truth = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}[time_series.index(curr_ts[i])]
            ground_truths.append(ground_truth)
            prompt = PROMPT_TEMPLATE.format(cap=cap, image_str=(IMAGE_STR if use_image else ""), ts1=time_series[0],
                                            ts2=time_series[1], ts3=time_series[2], ts4=time_series[3])
            all_prompts.append(prompt)
            all_plots.append([plot_paths[dataset][i]] + [plot_paths[dataset][z] for z in random_ts])

    all_prompts = list(zip(all_prompts, all_plots)) 
    write_prompts(all_prompts, ground_truths, out_dir, ts_paths, use_image)



if __name__ == "__main__":
    np.random.seed(414)
    main_local_data_hard(data_path="data/samples/new samples no overlap/test",
                    out_dir="ts_retrieval_cross_domain_with_image",
                    use_image=True)
