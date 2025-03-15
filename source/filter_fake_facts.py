import json
import os
from helpers import (
    save_file,
    filter_facts,
    load_config
)

"""FILTER_MODEL = "Google Gemini-2.0-Flash" #"OpenAI GPT-4o" #"Gemini-2.0-Flash"
FACTS_PATH = "/home/ubuntu/thesis/data/samples/captions/extracted facts" # where to look at
DATASET_NAMES = ["air quality", "border crossing", "crime", "demography"]#, "heart rate"]   
SAVE_PATH = "/home/ubuntu/thesis/data/samples/captions/filtered facts"
"""

def main():
    config = load_config()
    filder_model = config['model']['filter_model']
    facts_path = config['path']["extracted_facts_folder_path"]
    dataset_names = config['data']['dataset_names']
    save_folder_path = config['path']['filtered_facts_folder_path']


    for dataset_name in dataset_names:
        for filename in os.listdir(facts_path+"/"+dataset_name):
            if filename.startswith(dataset_name) and filename.endswith(".txt"):
                filepath = os.path.join(facts_path+"/"+dataset_name, filename)
                with open(filepath, 'r') as file:
                    facts = file.read()
                    filtered_facts = filter_facts(facts, model=filder_model)
                    
                filtered_facts = filtered_facts.split("\n")
                if filtered_facts[0].endswith(":"): # drop the first line if it ends with ":", as it's probably an introduction to the LLM's answer
                    filtered_facts = filtered_facts[1:]
                filtered_facts = "\n".join(filtered_facts)

                save_path = save_folder_path + f"/{dataset_name}/{filename[:-4]}_filtered.txt" 
                save_file(filtered_facts, save_path)
                print("\nSuccess: filtered facts from", filename)
                

if __name__ == "__main__":
    main()