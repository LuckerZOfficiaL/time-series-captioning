# This script expectes that sample_generation.py is run already and the captions are already generated. This script refines these captions.

import json
import os
from dataset_helpers import (
    save_file,
    add_facts_to_caption,
    change_linguistic_style,
    enrich_language,
    factual_checking
)

REFINEMENT_MODEL = "OpenAI GPT-4o" #"Gemini-2.0-Flash"
CHECKING_MODEL = "Gemini-1.5-Pro"
CAPTIONS_PATH = "/home/ubuntu/thesis/data/samples/captions" #/refined/add facts" # where to look for the captions to refine
DATASET_NAMES = ["air quality", "border crossing", "crime", "demography", "heart rate"]   
ASK_URLS = False #whether to ask the refinement model to provide URL references, even if it's True, the refiner doesn't give the URLs :C
REFINEMENT_TYPES = ["add facts", "change style", "enrich language", "factual checking"] # supported refinement types
REFINEMENT_TYPE = "change style" # "add facts", "change style", "enrich language", "factual checking"
DESIRED_STYLE = "academic" # used only if REFINEMENT_TYPE = "change style"
# Possible styles I can think of: casual, scientific, journalistic, technical, storytelling, academic

def main(dataset_names):
    if REFINEMENT_TYPE not in REFINEMENT_TYPES:
        print("\nAn erroneous refinement type is given!")
        exit()
    for dataset_name in dataset_names:
        # read all caption files from the folder, use the refinement model to add real facts and save them back into the original files
        for filename in os.listdir(CAPTIONS_PATH):
            if filename.startswith(dataset_name) and filename.endswith(".txt"): # and "refined" not in filename:
                filepath = os.path.join(CAPTIONS_PATH, filename)
                with open(filepath, 'r') as file:
                    caption = file.read()

                if REFINEMENT_TYPE == "add facts":
                    refined_caption = add_facts_to_caption(caption, model=REFINEMENT_MODEL, ask_urls=ASK_URLS)
                elif REFINEMENT_TYPE == "change style":
                    refined_caption = change_linguistic_style(caption, model=REFINEMENT_MODEL)
                elif REFINEMENT_TYPE == "enrich language":
                    refined_caption = enrich_language(caption, model=REFINEMENT_MODEL)
                elif REFINEMENT_TYPE == "factual checking":
                    refined_caption = factual_checking(caption, model=CHECKING_MODEL)

                save_folder = "/home/ubuntu/thesis/data/samples/captions/refined"
                if len(refined_caption) > int(0.7 * len(caption)) and caption not in refined_caption: # if the answer is much shorter than the original caption, the model has refused to refine the caption, so save only if that doesn't happen
                    if REFINEMENT_TYPE == "change style":
                        folder_path = os.path.join(save_folder, "change style", DESIRED_STYLE)
                        if not os.path.isdir(folder_path):  # Check if it's not a folder
                            os.makedirs(folder_path)  # Create the folder for that style
                        postfix = f"_{DESIRED_STYLE}.txt"
                        save_path = folder_path +"/" + filename[:-4] + postfix
                    elif REFINEMENT_TYPE == "enrich language":
                        postfix = f"_enriched.txt"
                        save_path = save_folder + "/enriched/" + filename[:-4] + postfix
                    elif REFINEMENT_TYPE == "add facts":
                        postfix = f"_facts.txt"
                        save_path = save_folder + "/add facts/" + filename[:-4] + postfix
                    elif REFINEMENT_TYPE == "factual checking":
                        postfix = f"_checked.txt"
                        save_path = save_folder + "/checked/" + filename[:-4] + postfix
                                                
                    save_file(refined_caption, save_path)
                    print("\nSuccess: refined caption with", REFINEMENT_TYPE, ":", filename)
                elif len(refined_caption) == len(caption) or caption in refined_caption: # if the model just copied the original caption
                    print("\nFailure: the Refinement model copied the original caption:", filename)
                else:
                    print("\nFailure: the Refinement Model refused to refine the caption:", filename)


if __name__ == "__main__":
    main(DATASET_NAMES)