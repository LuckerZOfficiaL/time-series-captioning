from functools import lru_cache
import json
import os
from pathlib import Path
import re

### Re-use imports from llava/eval/run_llava.py ###
import argparse
import torch

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
import re
### End re-use imports ###

from llava.eval.run_llava import image_parser, load_images 

from helpers import generate_prompt_for_baseline

#MODEL_PATH = "liuhaotian/llava-v1.6-34b"
MODEL_PATH = "liuhaotian/llava-v1.6-mistral-7b"
DATA_DIR = "/home/ubuntu/time-series-captioning/data/samples/"
OUT_DIR = "/home/ubuntu/time-series-captioning/llava_captions_mistral7b"

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
        "temperature": 0.3,  # NOTE: We set temperature to 0.3
        "top_p": None,
        "num_beams": 1,
        "max_new_tokens": 512
    })()
    return eval_llava(args)

@lru_cache
def _load_llava_model(model_path, model_base, model_name):
    return load_pretrained_model(
        model_path, model_base, model_name
    )

def eval_llava(args):
    """
    Taken from LLaVa source code `eval/run_llava.py`, modified
    not to reinstatiate the model on every inference.
    """
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = _load_llava_model(args.model_path, args.model_base, model_name)

    qs = args.query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image_files = image_parser(args)
    images = load_images(image_files)
    image_sizes = [x.size for x in images]
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    print(outputs)
    return outputs

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
    image_file = os.path.join(DATA_DIR, "plots_2.0", f"{ts_name}.jpeg")
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
