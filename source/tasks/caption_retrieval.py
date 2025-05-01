import numpy as np

from .task_helpers import run_prompt_creator

PROMPT_TEMPLATE = """
Here is a time series:
{ts}

{image_str}
What caption best describes to this time series?
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

def make_prompts(data):
    prompts = []
    for i, ts_data in enumerate(data):
        random_indices = np.random.randint(0, len(data), size=3)
        while i in random_indices:
            random_indices = np.random.randint(0, len(data), size=3)
        captions = [data[i]] + [data[z] for z in random_indices]
        np.random.shuffle(captions)
        ground_truth = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}[captions.index(ts_data)]
        prompt_no_image = PROMPT_TEMPLATE.format(ts=','.join([f"{x:.2f}" for x in ts_data["ts"]]), 
                                        image_str="", d1=captions[0], d2=captions[1], d3=captions[2], d4=captions[3])
        prompt_with_image = PROMPT_TEMPLATE.format(ts=','.join([f"{x:.2f}" for x in ts_data["ts"]]), 
                                        image_str=IMAGE_STR, d1=captions[0], d2=captions[1], d3=captions[2], d4=captions[3])
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
                       out_dir="caption_retrieval_same_domain",
                       groups=DATASETS)
    run_prompt_creator(make_prompts=make_prompts,
                       data_path="data/samples/new samples no overlap/test",
                       out_dir="caption_retrieval_cross_domain")
