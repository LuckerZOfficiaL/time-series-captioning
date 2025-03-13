import json
import os
import torch
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from helpers import (
    save_file,
    unify_facts
)

FACTS_PATH = "/home/ubuntu/thesis/data/samples/captions/filtered facts" # where to look at
DATASET_NAMES = ["air quality", "border crossing", "crime", "demography", "heart rate"]   
SAVE_PATH = "/home/ubuntu/thesis/data/fact bank"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SAVE_PCA = False


def embed_sentences(sentences, model_name="all-MiniLM-L6-v2"):
    """
    Embeds a list of sentences using a pretrained Sentence Transformer model.

    Args:
        sentences (list of str): The list of sentences to embed.
        model_name (str): The name of the Sentence Transformer model to use.

    Returns:
        torch.Tensor: A tensor of shape [N, embedding_size] containing the sentence embeddings.
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(sentences, convert_to_tensor=True)
    return embeddings

def save_embeddings_pca(sentences, model_name="all-MiniLM-L6-v2"):
    """
    Embeds sentences, performs PCA to reduce dimensionality to 2D, and visualizes them.

    Args:
        sentences (list of str): The list of sentences to embed.
        model_name (str): The name of the Sentence Transformer model to use.
    """
    # 1. Embed Sentences
    model = SentenceTransformer(model_name)
    embeddings = model.encode(sentences)  # No need for tensor here, PCA works with numpy

    # 2. Perform PCA
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)

    # 3. Visualize
    plt.figure(figsize=(10, 8))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1])

    # Add labels (optional)
    for i, sentence in enumerate(sentences):
        plt.annotate(str(i), (reduced_embeddings[i, 0], reduced_embeddings[i, 1]))  # Label with sentence index

    plt.title("Sentence Embeddings in 2D (PCA)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True)
    plt.savefig(SAVE_PATH+"/pca.jpeg")
    plt.close()



def main():
    all_facts_list = unify_facts(FACTS_PATH)
    save_file(all_facts_list, SAVE_PATH+"/all_facts.json")
    for i, fact in enumerate(all_facts_list):
        print(i, fact)

   
    all_facts_emb = embed_sentences(all_facts_list)
    save_file(all_facts_emb, SAVE_PATH+"/all_facts_emb.pth")
    print(all_facts_emb.shape)

    if SAVE_PCA: save_embeddings_pca(all_facts_list)

if __name__ == "__main__":
    main()