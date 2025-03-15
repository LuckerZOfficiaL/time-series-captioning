import json
import os
import random
from helpers import (
    save_file,
    extract_facts,
    load_config
)

"""EXTRACTION_MODEL = "Google Gemini-2.0-Flash" #"OpenAI GPT-4o" #"Gemini-2.0-Flash"
CAPTIONS_PATH = "/home/ubuntu/thesis/data/samples/captions/refined/add facts" # where to look for the captions to refine
DATASET_NAMES = ["air quality", "border crossing", "crime", "demography"]#, "heart rate"]   
SAVE_PATH = "/home/ubuntu/thesis/data/samples/captions/extracted facts"
"""

def main():
    config = load_config()
    random.seed(config['general']['random_seed'])

    dataset_names = config['data']['dataset_names']
    extraction_model = config['model']['extraction_model']
    look_at_captions_path = config['path']["refined_captions_folder_path"] + "/add facts"
    save_folder_path = config['path']['extracted_facts_folder_path']

    for dataset_name in dataset_names:
        for filename in os.listdir(look_at_captions_path):
            if filename.startswith(dataset_name) and filename.endswith(".txt"):
                filepath = os.path.join(look_at_captions_path, filename)
                with open(filepath, 'r') as file:
                    caption = file.read()
                    extracted_facts = extract_facts(caption, model=extraction_model)
                    
                    
                extracted_facts = extracted_facts.split("\n")
                if extracted_facts[0].endswith(":"): # drop the first line if it ends with ":", as it's probably an introduction to the LLM's answer
                    extracted_facts = extracted_facts[1:]
                extracted_facts = "\n".join(extracted_facts)

                save_path = save_folder_path + f"/{dataset_name}/{filename[:-4]}_facts.txt" 
                save_file(extracted_facts, save_path)
                print("\nSuccess: extracted facts from", filename)
                

if __name__ == "__main__":
    main()