import json
import os
from sentence_transformers import SentenceTransformer
from helpers import (
    save_file,
    embed_sentences,
    split_facts_by_time,
    load_config
)

"""
EMB_MODEL_NAME = "all-MiniLM-L6-v2"
BIN_YEARS = 10 # how long is a single period in years?
PERIODS_FACTS_PATH = f"/home/ubuntu/thesis/data/fact bank/by period/{BIN_YEARS}/all_facts_by_{BIN_YEARS}years.json"
SAVE_FOLDER = f"/home/ubuntu/thesis/data/fact bank/by period/{BIN_YEARS}"
"""

def main():
    config = load_config()
    emb_model_name = config['model']['embedding_model']
    bin_years = config['bank']['bin_years']
    periods_facts_path = f"{config['path']['all_facts_by_period_folder_path']}/{bin_years}/all_facts_by_{bin_years}years.json"
    save_folder = f"{config['path']['all_facts_by_period_folder_path']}/{bin_years}"

    with open(periods_facts_path.format(BIN_YEARS=bin_years)) as file:
        facts_by_period = json.load(file)
    
    embedding_model = SentenceTransformer(emb_model_name)

    for period in facts_by_period:
        period_facts_list = facts_by_period[period]
        if len(period_facts_list) > 0:
            period_folder = f"{save_folder}/{period}"
            os.makedirs(period_folder, exist_ok=True)
            
            save_file(period_facts_list, period_folder+"/facts_list.txt") # save facts
            
            facts_emb = embed_sentences(period_facts_list, model=embedding_model)
            save_file(facts_emb, period_folder+"/facts_emb.pth") # save embeddings

    print(f"\nSuccess: Saved facts and their embeddings by periods of {bin_years} years.")
            
if __name__ == "__main__":
    main()