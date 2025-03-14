import json
import os
from sentence_transformers import SentenceTransformer
from helpers import (
    save_file,
    embed_sentences,
    split_facts_by_time
)

EMB_MODEL_NAME = "all-MiniLM-L6-v2"
BIN_YEARS = 10 # how long is a single period in years?
PERIODS_FACTS_PATH = f"/home/ubuntu/thesis/data/fact bank/by period/{BIN_YEARS}/all_facts_by_{BIN_YEARS}years.json"
SAVE_FOLDER = f"/home/ubuntu/thesis/data/fact bank/by period/{BIN_YEARS}"

def main():
    with open(PERIODS_FACTS_PATH.format(BIN_YEARS=BIN_YEARS)) as file:
        facts_by_period = json.load(file)
    
    embedding_model = SentenceTransformer(EMB_MODEL_NAME)

    for period in facts_by_period:
        period_facts_list = facts_by_period[period]
        if len(period_facts_list) > 0:
            period_folder = f"{SAVE_FOLDER}/{period}"
            os.makedirs(period_folder, exist_ok=True)
            
            save_file(period_facts_list, period_folder+"/facts_list.txt") # save facts
            
            facts_emb = embed_sentences(period_facts_list, model=embedding_model)
            save_file(facts_emb, period_folder+"/facts_emb.pth") # save embeddings

    print("\nSuccess: Saved facts and their embeddings by period.")
            
if __name__ == "__main__":
    main()