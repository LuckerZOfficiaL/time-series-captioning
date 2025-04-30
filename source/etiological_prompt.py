from collections import defaultdict
import json
import pickle
import os

import numpy as np

DATA_PATH = "/home/ubuntu/reasoning_data/TS_Dataset.jsonl"

PROMPT_TEMPLATE = """
Here is a time series:
{ts}

{image_str}
What description best relates to this time series?
(A) {d1}
(B) {d2}
(C) {d3}
(D) {d4}

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

def main_local_data(data_path, out_dir, use_image=True):
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
    all_prompts = []
    ground_truths = []
    for i, ts in enumerate(all_ts):
        random_ts = np.random.randint(0, len(all_ts), size=3)
        while i in random_ts: 
            random_ts = np.random.randint(0, len(all_ts), size=3)
        captions = [all_captions[i]] + [all_captions[z] for z in random_ts]
        np.random.shuffle(captions)
        ground_truth = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}[captions.index(all_captions[i])]
        ground_truths.append(ground_truth)
        prompt = PROMPT_TEMPLATE.format(ts=','.join([f"{x:.2f}" for x in ts]), d1=captions[0],
                                        image_str=(IMAGE_STR if use_image else ""),
                                        d2=captions[1], d3=captions[2], d4=captions[3])
        all_prompts.append(prompt)
    plot_paths = [os.path.join(data_path, 'plots', path.replace(".txt", ".jpeg")) for path in caption_paths]
    all_prompts = list(zip(all_prompts, plot_paths)) 

    for prompt, gt, name in zip(all_prompts, ground_truths, ts_paths):
        prompt_text, plot_path = prompt
        json_entry = {"prompt": prompt_text}
        if use_image:
            json_entry["plot_path"] = plot_path
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
    all_prompts = []
    ground_truths = []
    for dataset, curr_captions in all_captions.items():
        curr_ts = all_ts[dataset]
        for i, ts in enumerate(curr_ts):
            random_ts = np.random.randint(0, len(curr_ts), size=3)
            while i in random_ts: 
                random_ts = np.random.randint(0, len(curr_ts), size=3)
            captions = [curr_captions[i]] + [curr_captions[z] for z in random_ts]
            np.random.shuffle(captions)
            ground_truth = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}[captions.index(curr_captions[i])]
            ground_truths.append(ground_truth)
            prompt = PROMPT_TEMPLATE.format(ts=','.join([f"{x:.2f}" for x in ts]), d1=captions[0],
                                            image_str=(IMAGE_STR if use_image else ""),
                                            d2=captions[1], d3=captions[2], d4=captions[3])
            all_prompts.append(prompt)
    plot_paths = [os.path.join(data_path, 'plots', path.replace(".txt", ".jpeg")) for path in caption_paths]
    all_prompts = list(zip(all_prompts, plot_paths)) 

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "prompts"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "ground_truth"), exist_ok=True)
    for prompt, gt, name in zip(all_prompts, ground_truths, ts_paths):
        prompt_text, plot_path = prompt
        json_entry = {"prompt": prompt_text}
        if use_image:
            json_entry["plot_path"] = plot_path
        with open(os.path.join(out_dir, "prompts", name.replace('.txt', '.json')), "w") as fh:
            json.dump(json_entry, fh) 
        with open(os.path.join(out_dir, "ground_truth", name), "w") as fh:
            fh.write(gt)  


def main_tsandlanguage():
    """
    Create test prompts from the external paper dataset
    https://github.com/behavioral-data/TSandLanguage
    """
    ts_data = load_data()
    all_prompts = []
    ground_truths = []
    for i, ts in enumerate(ts_data):
        random_ts = np.random.randint(0, len(ts_data), size=3)
        while i in random_ts: 
            random_ts = np.random.randint(0, len(ts_data), size=3)
        descriptions = [ts['description']] + [ts_data[z]['description'] for z in random_ts]
        np.random.shuffle(descriptions)
        ground_truth = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}[descriptions.index(ts['description'])]
        ground_truths.append(ground_truth)
        prompt = PROMPT_TEMPLATE.format(ts=','.join([f"{x:.2f}" for x in ts['series']]), d1=descriptions[0],
                                        d2=descriptions[1], d3=descriptions[2], d4=descriptions[3])
        all_prompts.append(prompt)
    print(all_prompts[0])

    with open("prompts_etiological.pkl", "wb") as fh:
        pickle.dump(all_prompts, fh)
    with open("gt_etiological.txt", 'w') as fh:
        for gt in ground_truths:
            fh.write(gt)
        

if __name__ == "__main__":
    np.random.seed(414)
    main_local_data_hard(data_path="data/samples/new samples no overlap/test",
                    out_dir="caption_retrieval_hard_no_image",
                    use_image=False)
