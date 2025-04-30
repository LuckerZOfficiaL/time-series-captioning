from collections import defaultdict
import json
import pickle
import os

import numpy as np

DATA_PATH = "/home/ubuntu/reasoning_data/TS_Dataset.jsonl"

PROMPT_TEMPLATE = """
Here is a time series:
{ts}

Here are four plots of different time series:
(A) {im_token}
(B) {im_token}
(C) {im_token}
(D) {im_token}
Which plot corresponds to the time series provided above? 

You must respond only with valid JSON, and no extra text or markdown.
The JSON schema is:
{{
  "answer": <string>
}}
<string> must be an answer string containing only A, B, C, or D.
Ensure your output parses as JSON with exactly one top-level object containing the answer field.
"""
IMAGE_STR = "I have also attached a line plot image of the time series to support you.\n"

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
    plot_paths = [os.path.join(data_path, 'plots', path.replace(".txt", ".jpeg")) for path in ts_paths]
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
    all_prompts = []
    ground_truths = []
    prompt_paths = []
    for i, ts in enumerate(all_ts):
        random_plots = np.random.randint(0, len(plot_paths), size=3)
        while i in random_plots: 
            random_plots = np.random.randint(0, len(plot_paths), size=3)
        plots = [plot_paths[i]] + [plot_paths[z] for z in random_plots]
        np.random.shuffle(plots)
        ground_truth = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}[plots.index(plot_paths[i])]
        ground_truths.append(ground_truth)
        ts_string = ','.join([f"{x:.2f}" for x in ts])
        prompt = PROMPT_TEMPLATE.format(ts=ts_string, im_token="<image>")
        all_prompts.append(prompt)
        prompt_paths.append(plots)
    import pdb; pdb.set_trace()

    all_prompts = list(zip(all_prompts, prompt_paths)) 
    write_prompts(all_prompts, ground_truths, out_dir, ts_paths, use_image)


def write_prompts(all_prompts, ground_truths, out_dir, ts_paths, use_image):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "prompts"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "ground_truth"), exist_ok=True)
    for prompt, gt, name in zip(all_prompts, ground_truths, ts_paths):
        prompt_text, plot_paths = prompt
        json_entry = {"prompt": prompt_text}
        if use_image:
            json_entry["plot_paths"] = plot_paths
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
    plot_paths = [os.path.join(data_path, 'plots', path.replace(".txt", ".jpeg")) for path in caption_paths]
    all_plots = defaultdict(list)
    for path in plot_paths:
        [dataset] = [d for d in DATASETS if d in path]
        all_plots[dataset].append(path)
    all_prompts = []
    ground_truths = []
    prompt_plots = []
    for dataset, curr_captions in all_captions.items():
        curr_ts = all_ts[dataset]
        curr_plots = all_plots[dataset]
        for i, ts in enumerate(curr_ts):
            random_plots = np.random.randint(0, len(curr_plots), size=3)
            while i in random_plots: 
                random_plots = np.random.randint(0, len(curr_plots), size=3)
            plots = [curr_plots[i]] + [curr_plots[z] for z in random_plots]
            np.random.shuffle(plots)
            ground_truth = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}[plots.index(curr_plots[i])]
            ground_truths.append(ground_truth)
            ts_string = ','.join([f"{x:.2f}" for x in ts])
            prompt = PROMPT_TEMPLATE.format(ts=ts, im_token="<image>") 
            all_prompts.append(prompt)
            prompt_plots.append(plots)

    all_prompts = list(zip(all_prompts, prompt_plots)) 
    write_prompts(all_prompts, ground_truths, out_dir, ts_paths, use_image)



if __name__ == "__main__":
    np.random.seed(414)
    main_local_data_hard(data_path="data/samples/new samples no overlap/test",
                    out_dir="plot_retrieval_hard",
                    use_image=True)
