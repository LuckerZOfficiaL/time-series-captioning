import json
import os
from dataset_helpers import (
    save_file,
    add_facts_to_caption
)

REFINEMENT_MODEL = "Gemini-1.5-Flash-Search"
captions_path = "/home/ubuntu/thesis/data/samples/captions"


def main(dataset_name):
    # read all caption files from the folder, use the refinement model to add real facts and save them back into the original files
    for filename in os.listdir(captions_path):
        if filename.startswith(dataset_name) and filename.endswith(".txt") and "refined" not in filename:
            filepath = os.path.join(captions_path, filename)
            with open(filepath, 'r') as file:
                caption = file.read()
            refined_caption = add_facts_to_caption(caption, model=REFINEMENT_MODEL)
                
            if len(refined_caption) > int(0.7 * len(caption)): # if the answer is much shorter than the original caption, the model has refused to refine the caption, so save only if that doesn't happen
                refined_filepath = filepath[:-4] + "_refined.txt"
                save_file(refined_caption, refined_filepath)
                print("\nSuccess: refined caption:", filename)
            elif len(refined_caption) == len(caption): # if no refinement is made
                print("\nFailure: the Refinement model copied the original caption:", filename)
            else:
                print("\nFailure: the Refinement Model refused to refine the caption:", filename)


if __name__ == "__main__":
    main("demography")