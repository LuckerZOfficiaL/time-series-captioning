import json
import os
import torch
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from helpers import (
    save_file,
    unify_facts,
    embed_sentences,
    save_embeddings_pca
)

FACTS_PATH = "/home/ubuntu/thesis/data/samples/captions/filtered facts" # where to look at
DATASET_NAMES = ["air quality", "border crossing", "crime", "demography"]#, "heart rate"]   
SAVE_PATH = "/home/ubuntu/thesis/data/fact bank"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SAVE_PCA = False



def main():
    all_facts_list = unify_facts(FACTS_PATH)
    save_file(all_facts_list, SAVE_PATH+"/all_facts.txt")
    for i, fact in enumerate(all_facts_list):
        print(i, fact)
   
    all_facts_emb = embed_sentences(all_facts_list, model_name=EMBEDDING_MODEL)
    save_file(all_facts_emb, SAVE_PATH+"/all_facts_emb.pth")
    print(all_facts_emb.shape)


    if SAVE_PCA: save_embeddings_pca(all_facts_list, model_name=EMBEDDING_MODEL)

if __name__ == "__main__":
    main()