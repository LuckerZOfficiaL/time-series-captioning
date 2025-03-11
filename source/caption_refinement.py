# This script expectes that sample_generation.py is run already and the captions are already generated. This script refines these captions.

import json
import os
from dataset_helpers import (
    save_file,
    add_facts_to_caption
)

REFINEMENT_MODEL = "OpenAI GPT-4o" #"Gemini-2.0-Flash"
CAPTIONS_PATH = "/home/ubuntu/thesis/data/samples/captions"
DATASET_NAMES = ["air quality", "border crossing", "crime", "demography", "heart rate"]   
ASK_URLS = False #whether to ask the refinement model to provide URL references, even if it's True, the refiner doesn't give the URLs :C

def main(dataset_names):
    for dataset_name in dataset_names:
        # read all caption files from the folder, use the refinement model to add real facts and save them back into the original files
        for filename in os.listdir(CAPTIONS_PATH):
            if filename.startswith(dataset_name) and filename.endswith(".txt") and "refined" not in filename:
                filepath = os.path.join(CAPTIONS_PATH, filename)
                with open(filepath, 'r') as file:
                    caption = file.read()
                refined_caption = add_facts_to_caption(caption, model=REFINEMENT_MODEL, ask_urls=ASK_URLS)
                    
                if len(refined_caption) > int(0.9 * len(caption)) and len(refined_caption) != len(caption): # if the answer is much shorter than the original caption, the model has refused to refine the caption, so save only if that doesn't happen
                    refined_filepath = filepath[:-4] + "_refined_urls.txt" if ASK_URLS else filepath[:-4] + "_refined.txt"
                    save_file(refined_caption, refined_filepath)
                    print("\nSuccess: refined caption:", filename)
                elif len(refined_caption) == len(caption): # if no refinement is made
                    print("\nFailure: the Refinement model copied the original caption:", filename)
                else:
                    print("\nFailure: the Refinement Model refused to refine the caption:", filename)


if __name__ == "__main__":
    main(DATASET_NAMES)