import json
import os
import random
from helpers import (
    save_file,
    extract_facts,
    load_config,
    check_single_fact,
    filter_sentences_no_non_year_numbers
)

"""EXTRACTION_MODEL = "Google Gemini-2.0-Flash" #"OpenAI GPT-4o" #"Gemini-2.0-Flash"
CAPTIONS_PATH = "/home/ubuntu/thesis/data/samples/captions/refined/add facts" # where to look for the captions to refine
DATASET_NAMES = ["air quality", "border crossing", "crime", "demography"]#, "heart rate"]   
SAVE_PATH = "/home/ubuntu/thesis/data/samples/captions/extracted facts"
"""

def main():
    config = load_config()
    random.seed(config['general']['random_seed'])
    words_to_skip = ['average', 'mean', 'standard deviation', 'above', 'below', 'all-time', 'increase','decline','decrease', 'series', 'fluctuations']

    dataset_names = config['data']['dataset_names']
    extraction_model = config['model']['extraction_model']
    checking_model = config['model']['checking_model']
    look_at_captions_path = config['path']["refined_captions_folder_path"] + "/add facts"
    save_folder_path = config['path']['verified_captions_folder_path']

    print(f"\nFiltering captions with falsities from {look_at_captions_path}...")
    for dataset_name in dataset_names:
        for filename in os.listdir(look_at_captions_path):
            if filename.startswith(dataset_name) and filename.endswith(".txt"):
                filepath = os.path.join(look_at_captions_path, filename)
                with open(filepath, 'r') as file:
                    caption = file.read()

                print("\nCaption: ", caption)
                extracted_facts = extract_facts(caption, model=extraction_model, return_list=True)
                extracted_facts = filter_sentences_no_non_year_numbers(extracted_facts)
                extracted_facts = [fact for fact in extracted_facts if not any(word in fact for word in words_to_skip)]
                print("\nExtracted facts: ", extracted_facts)

                is_true = True
                for fact in extracted_facts:
                    try:
                        outcome = check_single_fact(fact, checking_model=checking_model)
                        if outcome == False:
                            is_true = False
                            print(f"\nFalse: {fact}")
                            break
                        elif outcome is None:
                            print(f"\nInconclusive but we keep it: {fact}")   
                        else:
                            print(f"\nTrue: {fact}")  
                    except Exception as e:
                        print(f"\nGot Exception on fact:\n{fact} \n{e} ")
                        is_true = False
                        break
                                            
                
                if is_true == True:
                    save_path = save_folder_path + f"/{filename}" 
                    save_file(caption, save_path)
                    print(f"\n{filename} is verified and stored.")
                else:
                    print(f"\n{filename} is out due to falsity!")
                    
                                          
if __name__ == "__main__":
    main()