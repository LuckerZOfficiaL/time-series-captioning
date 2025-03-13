# This script expectes that sample_generation.py is run already and the captions are already generated. This script refines these captions.

import json
import os
from helpers import (
    save_file,
    extract_facts
)

EXTRACTION_MODEL = "Google Gemini-2.0-Flash" #"OpenAI GPT-4o" #"Gemini-2.0-Flash"
CAPTIONS_PATH = "/home/ubuntu/thesis/data/samples/captions/refined/add facts" # where to look for the captions to refine
DATASET_NAMES = ["air quality", "border crossing", "crime", "demography", "heart rate"]   
SAVE_PATH = "/home/ubuntu/thesis/data/samples/captions/extracted facts"

def main(dataset_names):
    for dataset_name in dataset_names:
        for filename in os.listdir(CAPTIONS_PATH):
            if filename.startswith(dataset_name) and filename.endswith(".txt"):
                filepath = os.path.join(CAPTIONS_PATH, filename)
                with open(filepath, 'r') as file:
                    caption = file.read()
                    extracted_facts = extract_facts(caption, model=EXTRACTION_MODEL)
                    
                    
                extracted_facts = extracted_facts.split("\n")
                if extracted_facts[0].endswith(":"): # drop the first line if it ends with ":", as it's probably an introduction to the LLM's answer
                    extracted_facts = extracted_facts[1:]
                extracted_facts = "\n".join(extracted_facts)

                save_path = SAVE_PATH + f"/{dataset_name}/{filename[:-4]}_facts.txt" 
                save_file(extracted_facts, save_path)
                print("\nSuccess: extracted facts from", filename)
                

if __name__ == "__main__":
    main(DATASET_NAMES)