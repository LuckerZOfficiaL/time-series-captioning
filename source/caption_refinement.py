# This script expectes that sample_generation.py is run already and the captions are already generated. This script refines these captions.

import json
import os
import random
from helpers import (
    save_file,
    add_facts_to_caption,
    change_linguistic_style,
    enrich_language,
    #factual_checking,
    refine_caption_with_corrected_facts,
    load_config
)

"""REFINEMENT_MODEL = "Google Gemini-2.0-Flash" #"OpenAI GPT-4o" #"Gemini-2.0-Flash"
CHECKING_MODEL = "Google Gemini-2.0-Flash"
CAPTIONS_PATH = "/home/ubuntu/thesis/data/samples/captions/refined/add facts" # where to look for the captions to refine
DATASET_NAMES = ["air quality", "border crossing", "crime", "demography"]#, "heart rate"]   
ASK_URLS = False #whether to ask the refinement model to provide URL references, even if it's True, the refiner doesn't give the URLs :C
REFINEMENT_TYPES = ["add facts", "change style", "enrich language", "factual checking"] # supported refinement types
REFINEMENT_TYPE = "factual checking" # "add facts", "change style", "enrich language", "factual checking"
DESIRED_STYLE = "academic" # used only if REFINEMENT_TYPE = "change style"
# Possible styles I can think of: casual, scientific, journalistic, technical, storytelling, academic"""

def main():
    config = load_config()
    random.seed(config['general']['random_seed'])
    refinement_model = config['model']['refinement_model']
    checking_model = config['model']['checking_model']
    captions_folder_path = config['path']['captions_folder_path']
    dataset_names = config['data']['dataset_names']
    refinement_types = config['refinement']['refinement_types']
    refinement_type = config['refinement']['refinement_type']
    desired_style = config['refinement']['desired_style']
    refinement_look_at_folder = config['refinement']["refinement_target"]

    look_at_captions_path = captions_folder_path + "/" + refinement_look_at_folder
    print("\nRefining captions from folder", refinement_look_at_folder)


    if refinement_type not in refinement_types:
        print("\nAn erroneous refinement type is given!")
        exit()
    for dataset_name in dataset_names:
        # read all caption files from the folder, use the refinement model to add real facts and save them back into the original files
        for filename in os.listdir(look_at_captions_path):
            if filename.startswith(dataset_name) and filename.endswith(".txt"): # and "refined" not in filename:
                filepath = os.path.join(look_at_captions_path, filename)
                with open(filepath, 'r') as file:
                    caption = file.read()

                if refinement_type == "add facts":
                    refined_caption = add_facts_to_caption(caption, model=refinement_model)
                    if len(refined_caption.split('\n\n')) > 1: # if there are more than one paragraph, discard the first as it is likely an introduction to the answer from the refiner model
                            paragraphs = refined_caption.split('\n\n')
                            refined_caption = paragraphs[1:]
                        
                elif refinement_type == "change style":
                    refined_caption = change_linguistic_style(caption, model=refinement_model)
                elif refinement_type == "enrich language":
                    refined_caption = enrich_language(caption, model=refinement_model)
                elif refinement_type == "factual checking":
                    #refined_caption = factual_checking(caption, model=checking_model)
                    refined_caption = refine_caption_with_corrected_facts(caption, 
                                        model=refinement_model,
                                        correction_method=config["refinement"]['factual_correction_method'],
                                        skip_numeric=config['refinement']['skip_numeric'])

                save_folder = config['path']['refined_captions_folder_path'] #"/home/ubuntu/thesis/data/samples/captions/refined"
                if len(refined_caption) > int(0.7 * len(caption)) and caption not in refined_caption: # if the answer is much shorter than the original caption, assume the model has refused to refine the caption, so save only if that doesn't happen
                    if refinement_type == "change style":
                        folder_path = os.path.join(save_folder, "change style", desired_style)
                        if not os.path.isdir(folder_path):  # Check if it's not a folder
                            os.makedirs(folder_path)  # Create the folder for that style
                        postfix = f"_{desired_style}.txt"
                        save_path = folder_path +"/" + filename[:-4] + postfix
                    elif refinement_type == "enrich language":
                        postfix = f"_enriched.txt"
                        save_path = save_folder + "/enriched/" + filename[:-4] + postfix
                    elif refinement_type == "add facts":
                        postfix = f"_concretized.txt"
                        save_path = save_folder + "/add facts/" + filename[:-4] + postfix
                    elif refinement_type == "factual checking":
                        postfix = f"_checked.txt"
                        save_path = save_folder + "/checked/" + filename[:-4] + postfix
                                                
                    save_file(refined_caption, save_path)
                    print(f"\nSuccess: {refinement_model} refined caption with", refinement_type, ":", filename)
                elif len(refined_caption) == len(caption) or caption in refined_caption: # if the model just copied the original caption
                    print(f"\nFailure: {refinement_model} copied the original caption:", filename)
                else:
                    print(f"\nFailure: {refinement_model} refused to refine the caption:", filename)


if __name__ == "__main__":
    main()