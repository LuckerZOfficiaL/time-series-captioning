from functools import lru_cache
import json
import os
from pathlib import Path
import re
import pickle
import time

from datasets import Dataset
import numpy as np
from numpy.core.multiarray import _reconstruct
import requests
from PIL import Image
from qwen_omni_utils import process_mm_info
import torch
from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer, TrainingArguments
from trl import SFTConfig, SFTTrainer

from helpers import generate_prompt_for_baseline

MODEL_PATH = "Qwen/Qwen2.5-Omni-7B"
DATA_DIR = "/home/ubuntu/time-series-captioning/data/samples/new samples no overlap/train"
OUT_DIR = "/home/ubuntu/time-series-captioning/qwen_fine_tune_2"


@lru_cache
def _load_batch_qwen_model(model_name, device):
    torch.manual_seed(314)
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-Omni-7B",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,        # now picks up the patched forward()
    )
    model.config.vocab_size = model.config.text_config.vocab_size  # hacky patch
    model.to(device)
    processor = Qwen2_5OmniProcessor.from_pretrained(model_name)
    return model, processor

def format_conversation(prompt, image_file, label, processor):
    conversation = [
        {
            "role": "system",
            "content": [{"type": "text", "text": ("You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group,"
                       " capable of perceiving auditory and visual inputs, as well as generating text and speech.")}]
        },
        {
            "role": "user",
            "content": [{"type": "image", "image": image_file},
                        {"type": "text", "text": prompt}]
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": label}]
        }
    ]
    return conversation

def get_train_dataset(data_dir, processor):
    ts_dir = os.path.join(data_dir, "time series")
    ts_names = [Path(fn).stem for fn in os.listdir(ts_dir)]
    prompts = []
    image_files = []
    labels = []
    print(f"Loading time series dataset, length {len(ts_names)}")
    for ts_name in ts_names:
        dataset_name = ts_name.split("_")[0]
        metadata_path = os.path.join(data_dir, "metadata", f"{ts_name}.json")
        ts_file_path = os.path.join(data_dir, "time series", f"{ts_name}.txt")
        label_file_path = os.path.join(data_dir, "gt_captions", f"{ts_name}.txt")
        with open(metadata_path, 'r') as fh:
            metadata = json.load(fh)
        with open(ts_file_path, 'r') as fh:
            ts = fh.read()
        with open(label_file_path, 'r') as fh:
            labels.append(fh.read())
        prompt = generate_prompt_for_baseline(dataset_name, metadata, ts, use_image=True)
        image_file = os.path.join(data_dir, "plots", f"{ts_name}.jpeg")
        prompts.append(prompt)
        image_files.append(image_file)
    conversations = [format_conversation(p, i, l, processor) for p, i, l in zip(prompts, image_files, labels)]
    conversations = processor.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)
    return [{"chat": c, "image": i} for c, i in zip(conversations, image_files)]

def eval_batch_qwen(prompts, image_files, device, use_image): 
    model, processor = _load_batch_qwen_model(MODEL_PATH, device)
    inputs = processor(text=text, images=images, return_tensors="pt", padding=True, use_audio_in_video=False)
    inputs = inputs.to(model.device).to(model.dtype)

    # Batch Inference
    stime = time.time()
    text_ids = model.generate(**inputs, max_new_tokens=256, temperature=0.3, do_sample=True, 
                              use_audio_in_video=False, return_audio=False)
    print(f"RUNTIME on {device}: {time.time() - stime:.2f} seconds")
    trimmed_generated_ids = [out_ids[len(in_ids) :] for in_ids, out_ids in 
                             zip(model_inputs.input_ids, generated_ids)]
    captions = processor.batch_decode(trimmed_generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return captions

DEVICE = 'cuda'

def collate_fn(examples):
    _, processor = _load_batch_qwen_model(MODEL_PATH, DEVICE)
    text = [ex["chat"] for ex in examples]
    #_, images, _ = process_mm_info(examples)
    images = [Image.open(ex["image"]) for ex in examples]
    # Tokenize the texts and process the images
    batch = processor(
        text=text, images=images, return_tensors="pt", padding=True
    ) 
    # The labels are the input_ids, and we mask the padding tokens in the loss computation
    labels = batch["input_ids"].clone()  # Clone input IDs for labels
    labels[labels == processor.tokenizer.pad_token_id] = -100  # Mask padding tokens in labels
    # Ignore the image token index in the loss computation 
    # Mask image token IDs in the labels
    image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]  # Convert image token to ID
    for image_token_id in image_tokens:
        labels[labels == image_token_id] = -100  # Mask image token IDs in labels

    batch["labels"] = labels  # Add labels to the batch

    return batch  # Return the prepared batch

def debug_batch(batch):
    # gpt test stuff
    labels = batch["labels"]
    valid_tokens = (labels != -100).sum()
    print(f"  Valid label tokens in this batch: {valid_tokens.item()}")
    # After forward:
    outputs = model(**batch)
    logits = outputs.logits
    print("  Any NaNs in logits? ", torch.isnan(logits).any().item())
    loss = loss_fn(logits=logits, labels=labels, vocab_size=model.config.vocab_size)
    print("  Raw loss:", loss.item(), "NaN?", torch.isnan(loss).item())
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    print("  Grad‑norm (clipped):", gn)


def main(model_eval, data_dir, out_dir, use_image=True):
    model, processor = _load_batch_qwen_model(MODEL_PATH, DEVICE)
    model.gradient_checkpointing_enable()  # decrease GPU mem usage
    #model.gradient_checkpointing_disable()
    #training_data = get_train_dataset(data_dir, processor)
    with open("training_data.pkl", "rb") as fh:
        training_data = pickle.load(fh)

    train_dataset = Dataset.from_list(training_data)
    
    torch.autograd.set_detect_anomaly(True) 

    training_args = SFTConfig(
        output_dir=out_dir,
        num_train_epochs=1,
        learning_rate=3e-8,
        warmup_steps=100,
        max_grad_norm=1.0,
        lr_scheduler_type="inverse_sqrt",
        optim="adamw_torch_fused",
        remove_unused_columns=False,
        per_device_train_batch_size=1, # decrease GPU mem usage
        #gradient_accumulation_steps=4, # decrease GPU mem usage 
        logging_strategy="steps",
        logging_steps=1,
        report_to="wandb",
        log_level="info",
        dataset_text_field="",
        save_strategy="steps",
        save_steps=500,
        save_total_limit=1,
    )
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}
    import wandb 
    wandb.init(
        project="qwen25-fine-tuning",  
        name="qwen25-fine-tuning",  
        config=training_args,
    )
 
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
    )

    
    print("Now training model")
    trainer.train(resume_from_checkpoint=True)
    print("Model training complete, now saving")
    trainer.save_model(training_args.output_dir)

    
if __name__ == "__main__":
    main(eval_batch_qwen, DATA_DIR, OUT_DIR, use_image=True)
