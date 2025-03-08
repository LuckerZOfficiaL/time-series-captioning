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
        if filename.startswith(dataset_name) and filename.endswith(".txt"):
            filepath = os.path.join(captions_path, filename)
            with open(filepath, 'r') as file:
                caption = file.read()
            refined_caption = add_facts_to_caption(caption, model=REFINEMENT_MODEL)
            save_file(refined_caption, filepath)


if __name__ == "__main__":
    main("crime")