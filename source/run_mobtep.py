import sys
import os
import torch
import json
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '/home/ubuntu/thesis/model')))
from mobtep import Mobtep
from visual_encoder import preprocess_image

from helpers import (
    load_config,
    read_txt_to_num_list,
    read_jpeg_to_tensor,
    read_txt_to_string
)




def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config()
    random.seed(config['general']['random_seed'])
    
    ts_folder_path = config['path']['ts_folder_path']
    caption_folder_path = config['path']['caption_folder_path'] + "/raw"
    plot_folder_path = config['path']['plot_folder_path']
    #dataset_names = config['data']['dataset_names']


    with open(config['path']['prototypes_path'], 'r') as file:
        prototype_words = file.read().split(',')
    prototype_words = [word.strip() for word in prototype_words]
    

    mobtep = Mobtep(tcn_emb_size=config['mobtep']['tcn_emb_size'], 
                    prototype_words=prototype_words, 
                    use_linear_proj=config['mobtep']['use_linear_proj']).to(device)
    mobtep.eval()

    sample_names = os.listdir(ts_folder_path)
    sample_names = [sample_name.split('.')[0] for sample_name in sample_names] # get the filename ignoring the extension 
    ts_input = []

    for sample_name in sample_names:
        print("\nSample ", sample_name)
        ts_path = os.path.join(ts_folder_path, f"{sample_name}.txt")
        caption_path = os.path.join(caption_folder_path, f"{sample_name}.txt")
        plot_path = os.path.join(plot_folder_path, f"{sample_name}.jpeg")
        
        caption = read_txt_to_string(caption_path)

        ts = read_txt_to_num_list(ts_path)
        ts = torch.tensor(ts, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device) # make the list a valid tensor and send it to device
        
        prompt = ["Please describe this time series."]
        
        plot_tensor = preprocess_image(plot_path).to(device)
        
        #print(f"TS shape: {ts.shape}, plot shape: {plot_tensor.shape}")


        print("\nTS: ", ts)
        
        with torch.no_grad():
            output = mobtep(ts, prompt, plot_tensor, output_text=True)

        for i, caption in enumerate(output):
            print(f"\n\n{i})\n", caption)

        break

    #ts_input = torch.randn(3, 100, 1).to(device)  # Example time series (3 samples, length 100, 1 channel)
    #text_input = ["Tell me a story.", "How are you?", "I am Luca."]
    #visual_input = torch.randn(3, 3, 224, 224).to(device)  # Example visual data (3 samples, 3 channels, H224 x W224 images)

if __name__ == "__main__":
    main()
    

