import json
import os
from pathlib import Path

from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model

from helpers import generate_prompt_for_baseline

MODEL_PATH = "liuhaotian/llava-v1.6-34b"
DATA_DIR = "/home/ubuntu/time-series-captioning/data/samples/"
OUT_DIR = "/home/ubuntu/time-series-captioning/llava_captions"

def _eval_model(prompt, image_file):
    """
    Evaluate desired model for given prompt and image 
    """
    args = type('Args', (), {
        "model_path": MODEL_PATH,
        "model_base": None,
        "model_name": get_model_name_from_path(MODEL_PATH),
        "query": prompt,
        "conv_mode": None,
        "image_file": image_file,
        "sep": ",",
        "temperature": 0,
        "top_p": None,
        "num_beams": 1,
        "max_new_tokens": 512
    })()
    return eval_model(args)

def write_caption(ts_name):
    """
    Given ts_name, write a caption .txt file for the time series.
    """
    dataset_name = ts_name.split("_")[0]
    with open(os.path.join(DATA_DIR, "metadata", f"{ts_name}.json"), 'r') as fh:
        metadata = json.load(fh)
    with open(os.path.join(DATA_DIR, "time series", f"{ts_name}.txt"), 'r') as fh:
        ts = fh.read()
    prompt = generate_prompt_for_baseline(dataset_name, metadata, ts)
    image_file = os.path.join(DATA_DIR, "plots", f"{ts_name}.jpeg")
    caption = _eval_model(prompt, image_file)
    with open(os.path.join(OUT_DIR, f"{ts_name}.txt"), "w+") as fh:
        fh.write(caption)


def main():
    ts_names = [Path(fn).stem for fn in os.listdir(os.path.join(DATA_DIR, "time series"))]
    done_names = {Path(fn).stem for fn in os.listdir(OUT_DIR)}
    ts_names = sorted([name for name in ts_names if name not in done_names])
    for ts_name in ts_names:
        print(f"Writing caption for {ts_name}")
        write_caption(ts_name)

if __name__ == "__main__":
    main()
