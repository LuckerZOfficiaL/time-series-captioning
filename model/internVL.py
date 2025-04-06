import torch
from transformers import AutoTokenizer, AutoModel
from PIL import Image
import torchvision.transforms as T
from helpers import(
    load_config
)
from custom_methods import(
    custom_generate
)
import types

# Image transformation
def transform_image(image_path, input_size=448):
    transform = T.Compose([
        T.Resize((input_size, input_size)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image)

# Load batch of images
def load_batch(image_paths, input_size=448):
    images = [transform_image(img_path, input_size) for img_path in image_paths]
    return torch.stack(images).to(torch.bfloat16).cuda() # type: ignore

# Batch inference
def batch_inference(model, image_paths, prompts, tokenizer=None, ts_emb=None, max_output_tokens=256):
    pixel_values = load_batch(image_paths)
    num_patches_list = [1] * len(image_paths)  # Assuming each image is one patch
    if tokenizer is None:
        responses = model.batch_chat(
            pixel_values=pixel_values, ts_emb=ts_emb, num_patches_list=num_patches_list,
            questions=prompts, generation_config={'max_new_tokens': max_output_tokens, 'do_sample': True}
        )
    else:
        responses = model.batch_chat(
            tokenizer=tokenizer, pixel_values=pixel_values, ts_emb=ts_emb, num_patches_list=num_patches_list,
            questions=prompts, generation_config={'max_new_tokens': max_output_tokens, 'do_sample': True}
        )
    return responses

# this function is actually for Mob, even though it is in this script
def mob_batch_inference(model, image_paths, prompts, ts, tokenizer=None, max_output_tokens=256):
    pixel_values = load_batch(image_paths)
    num_patches_list = [1] * len(image_paths)  # Assuming each image is one patch
    if tokenizer is None:
        responses = model.batch_chat(
            pixel_values=pixel_values, ts=ts, num_patches_list=num_patches_list,
            questions=prompts, generation_config={'max_new_tokens': max_output_tokens, 'do_sample': True}
        )
    else:
        responses = model.batch_chat(
            tokenizer=tokenizer, pixel_values=pixel_values, ts=ts, num_patches_list=num_patches_list,
            questions=prompts, generation_config={'max_new_tokens': max_output_tokens, 'do_sample': True}
        )
    return responses

def main():
    config = load_config()
    path = config['mobtep']['internvl_name']
    model = AutoModel.from_pretrained( 
        path,
        torch_dtype=torch.bfloat16,
        #low_cpu_mem_usage=True,
        #use_flash_attn=True,
        trust_remote_code=True).eval().cuda()
    
    model.generate = types.MethodType(custom_generate, model)
    
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

    image_paths = ['/home/ubuntu/thesis/data/samples/plots/air quality_0.jpeg',
                '/home/ubuntu/thesis/data/samples/plots/demography_0.jpeg']

    prompts = ['Describe this line chart about the hourly CO levels in London. Discuss the values you see.', 
                'Describe this line chart about the yearly death rates in Greece. Discuss the values you see.']

    responses = batch_inference(model, tokenizer, image_paths, prompts)

    for prompt, response in zip(prompts, responses):
        print(f'\nUser: {prompt}\nAssistant: {response}\n')


if __name__ == "__main__":
    main()
