import json
import os
import torch
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from helpers import (
    save_file,
    unify_facts,
    embed_sentences,
    save_embeddings_pca,
    load_config
)

"""FACTS_PATH = "/home/ubuntu/thesis/data/samples/captions/filtered facts" # where to look at
DATASET_NAMES = ["air quality", "border crossing", "crime", "demography"]#, "heart rate"]   
SAVE_PATH = "/home/ubuntu/thesis/data/fact bank"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
SAVE_PCA = True"""



def main():
    config = load_config()
    facts_path = config['path']['filtered_facts_folder_path'] # where to look for facts
    dataset_names = config['data']['dataset_names']
    save_folder_path = config['path']["fact_bank_folder_path"]
    embedding_model_name = config['model']['embedding_model']
    save_pca = config['bank']['save_pca']
    


    embedding_model = SentenceTransformer(embedding_model_name)
    all_facts_list = unify_facts(facts_path)
    save_file(all_facts_list, save_folder_path+"/all_facts.txt")
    for i, fact in enumerate(all_facts_list):
        print(i, fact)
   
    all_facts_emb = embed_sentences(all_facts_list, model=embedding_model)

    save_file(all_facts_emb, save_folder_path+"/all_facts_emb.pth")
    print(all_facts_emb.shape)


    if save_pca: 
        print("\nGenerating PCA...")
        save_embeddings_pca(all_facts_list, model=embedding_model, save_path=save_folder_path+"/pca.jpeg")

if __name__ == "__main__":
    main()