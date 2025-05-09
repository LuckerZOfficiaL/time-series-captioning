import copy

import numpy as np

from .task_helpers import run_prompt_creator

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


def make_prompts(data):
    prompts = []
    for i, ts_data in enumerate(data): 
        # PERTURBATIONS: reverse, shuffle, noise with std_dev 0.01 * abs(x)
        ts = ts_data["ts"]
        reversed_ts = list(reversed(ts))
        shuffled_ts = copy.deepcopy(ts)
        np.random.shuffle(shuffled_ts)
        # Round the noised number to the same significant figures as used in the original number
        noised_ts = [round(x + np.random.normal(loc=0, scale=(0.01 * abs(x))), 
                           len(str(x).split('.')[1].replace('-', '')) + 1) for x in ts] 

        time_series = [ts, reversed_ts, shuffled_ts, noised_ts]
        np.random.shuffle(time_series)
        ground_truth = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}[time_series.index(ts_data["ts"])]
        prompt_no_image = PROMPT_TEMPLATE.format(cap=ts_data["caption"], image_str="", ts1=time_series[0],
                                        ts2=time_series[1], ts3=time_series[2], ts4=time_series[3])
        prompts.append({
            "ts_name": ts_data["ts_name"],
            "prompt_no_image": prompt_no_image,
            "ground_truth": ground_truth
        })
    import pprint
    pprint.pprint(prompts[0])
    return prompts


if __name__ == "__main__":
    np.random.seed(414)
    DATASETS = ["air quality", "crime", "border crossing", "demography", "road injuries", "covid",
                "co2", "diet", "online retail", "walmart", "agriculture"]
    run_prompt_creator(make_prompts=make_prompts,
                       data_path="data/samples/new samples no overlap/test",
                       out_dir="perturbed_ts_matching")
