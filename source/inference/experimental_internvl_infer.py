import os
import json
import torch
import multiprocessing as mp
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
# from transformers import AutoTokenizer
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
# import warnings
import sys
# from transformers import AutoModelForCausalLM
from helper import generate_prompt_for_baseline

# the custom model path 
sys.path.append("/home/ubuntu/projects/time_series_main/models/internvl2_5_2b")
# from modeling_internvl_chat import InternVLChatModel

# processing image --- 
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff and area > 0.5 * image_size**2 * ratio[0] * ratio[1]:
            best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = sorted(
        {(i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1)
         if 1 <= i * j <= max_num},
        key=lambda x: x[0] * x[1]
    )
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    return torch.stack(pixel_values)


def collate_fn(batch):
    return list(zip(*batch))

def process_chunk(gpu_id, chunk, data_dir, output_dir, dataset_name):
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"

    from transformers import AutoTokenizer
    from modeling_internvl_chat import InternVLChatModel

    base_model_path = "/home/ubuntu/projects/time_series_main/models/internvl2_5_2b"
    finetuned_checkpoint = "/home/ubuntu/projects/time_series_main/outputs/tsqa_finetune_filter"

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    model = InternVLChatModel.from_pretrained(
        base_model_path,
        trust_remote_code=True,
    ).to(device)

    from safetensors.torch import load_file as load_safetensors
    model.load_state_dict(load_safetensors(f"{finetuned_checkpoint}/model.safetensors", device=device), strict=False)     # load the LORA + finetuned weights 
    model.eval()
    model = model.half() 
    
    dataset = TimeSeriesDataset(chunk, data_dir, dataset_name)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    generation_config = {"num_beams": 1, "max_new_tokens": 256, "do_sample": False}

    for names, metadatas, tss, pixel_values_list in tqdm(loader, desc=f"GPU {gpu_id}", position=gpu_id):
        try:
            name = names[0]
            metadata = metadatas[0]
            ts = tss[0]
            pixel_values = pixel_values_list[0].to(torch.float16).to(device)
            prompt = generate_prompt_for_baseline(dataset_name, metadata, ts)
            question = f"<image>\n{prompt.strip()}"
            caption = model.chat(tokenizer, pixel_values, question, generation_config).strip()
            # print(f"GPU {gpu_id} {name} => {caption[:80]}")
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, f"{name}.txt"), "w", encoding="utf-8") as f:
                f.write(caption)
        except Exception as e:
            tqdm.write(f"GPU {gpu_id} failed {name} — {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    dataset_names = [
        "air quality", "crime", "border crossing", "demography", "road injuries", "covid", 
        "co2", "diet", "online retail", "walmart", "agriculture"
    ]

    data_dir = "/home/ubuntu/projects/new_data/test"
    output_base = "/home/ubuntu/projects/outputs1/internvl-finetune-filter"

    for dataset_name in dataset_names:
        captions_dir = os.path.join(data_dir, "captions")
        names = sorted([
            f.replace(".txt", "")
            for f in os.listdir(captions_dir)
            if f.startswith(f"{dataset_name}_") 
        ])

        print(f"{len(names)} samples for dataset: {dataset_name}")
        if not names:
            continue

        gpu_ids = [0,2,7]
        chunks = [names[i::len(gpu_ids)] for i in range(len(gpu_ids))]

        processes = []
        for i, gpu_id in enumerate(gpu_ids):
            output_dir = output_base
            p = mp.Process(target=process_chunk, args=(gpu_id, chunks[i], data_dir, output_dir, dataset_name))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
