import numpy as np

from .task_helpers import run_prompt_creator

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


def make_prompts(data):
    prompts = []
    for i, ts_data in enumerate(data): 
        random_indices = np.random.randint(0, len(data), size=3)
        while i in random_indices: 
            random_indices = np.random.randint(0, len(data), size=3)
        time_series = [data[i]] + [data[z] for z in random_indices]
        np.random.shuffle(time_series)
        ground_truth = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}[time_series.index(ts_data)]
        ts_strings = [','.join([f"{x:.2f}" for x in ts["ts"]]) for ts in time_series]
        prompt_no_image = PROMPT_TEMPLATE.format(cap=ts_data["caption"], image_str="", ts1=time_series[0]["ts"],
                                        ts2=time_series[1]["ts"], ts3=time_series[2]["ts"], ts4=time_series[3]["ts"])
        prompt_with_image = PROMPT_TEMPLATE.format(cap=ts_data["caption"], image_str=IMAGE_STR, ts1=time_series[0]["ts"],
                                        ts2=time_series[1]["ts"], ts3=time_series[2]["ts"], ts4=time_series[3]["ts"])
        image_paths = [ts_data["plot"]] + [data[z]["plot"] for z in random_indices]
        prompts.append({
            "ts_name": ts_data["ts_name"],
            "prompt_no_image": prompt_no_image,
            "prompt_with_image": prompt_with_image,
            "image_paths": image_paths,
            "ground_truth": ground_truth
        })
    return prompts


if __name__ == "__main__":
    np.random.seed(414)
    DATASETS = ["air quality", "crime", "border crossing", "demography", "road injuries", "covid",
                "co2", "diet", "online retail", "walmart", "agriculture"]
    run_prompt_creator(make_prompts=make_prompts,
                       data_path="data/samples/new samples no overlap/test",
                       out_dir="ts_retrieval_same_domain",
                       groups=DATASETS)
    run_prompt_creator(make_prompts=make_prompts,
                       data_path="data/samples/new samples no overlap/test",
                       out_dir="ts_retrieval_cross_domain")
