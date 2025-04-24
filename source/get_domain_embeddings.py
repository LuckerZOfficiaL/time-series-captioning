import os
import torch
from sentence_transformers import SentenceTransformer
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from torch.nn.functional import cosine_similarity
import glob
import numpy as np



def get_domain_embeddings():
    captions_folder = "/home/ubuntu/thesis/data/samples/new samples no overlap/train/gt_captions"
    removed_domains = ["co2", "demography", "agriculture"]

    model = SentenceTransformer("all-mpnet-base-v2")  # You can switch to any other sBERT model

    # Group files by domain
    file_paths_per_domain = defaultdict(list)
    for filename in os.listdir(captions_folder):
        domain = filename.split("_")[0]
        file_paths_per_domain[domain].append(filename)
        if domain not in removed_domains:
            file_paths_per_domain["filtered_train_domains"].append(filename)
            

    
    
    embeddings_per_domain = {}

    for domain, filenames in file_paths_per_domain.items():
        print(f"Processing domain: {domain} with {len(filenames)} files")

        texts = []
        for filename in filenames:
            with open(os.path.join(captions_folder, filename), 'r') as file:
                texts.append(file.read())

        # Get sentence embeddings
        domain_embeddings = model.encode(texts, batch_size=32, convert_to_numpy=True, show_progress_bar=True)

        # Compute mean embedding
        domain_mean_embedding = np.mean(domain_embeddings, axis=0, keepdims=True)
        embeddings_per_domain[domain] = domain_mean_embedding

        print(f"Completed domain: {domain}, embedding shape: {domain_mean_embedding.shape}")
  
        save_path = os.path.join("/home/ubuntu/thesis/source/domain embeddings", f"{domain}_embedding.pth")
        torch.save(torch.tensor(domain_mean_embedding), save_path)
        
        print(f"Saved embedding for domain: {domain} to {save_path}")

    #return embeddings_per_domain


def main():
    get_domain_embeddings()
    
    embeddings_folder = "/home/ubuntu/thesis/source/domain embeddings"
    domain_embeddings = {}
    for file_path in glob.glob(os.path.join(embeddings_folder, "*.pth")):
        domain = os.path.basename(file_path).replace("_embedding.pth", "")
        domain_embeddings[domain] = torch.load(file_path)

    print("Embeddings loaded.")
    
    # Calculate pairwise cosine similarity
    domains = list(domain_embeddings.keys())
    num_domains = len(domains)
    similarity_matrix = torch.zeros((num_domains, num_domains))

    for i, domain1 in enumerate(domains):
        for j, domain2 in enumerate(domains):
            if i <= j:  # Compute only for upper triangle and diagonal
                sim = cosine_similarity(
                    domain_embeddings[domain1], domain_embeddings[domain2]
                )
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim  # Symmetric matrix

    # Create heatmap
    plt.figure(figsize=(15, 12))
    sns.heatmap(
        similarity_matrix.numpy(),
        xticklabels=domains,
        yticklabels=domains,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
    )
    plt.title("Pairwise Cosine Similarity Between Domains")
    plt.tight_layout()

    # Save the plot
    plot_path = "/home/ubuntu/thesis/source/domain embeddings/domain_similarity_heatmap.jpg"
    plt.savefig(plot_path)
    print(f"Heatmap saved to {plot_path}")
    
    
if __name__ == "__main__":
    main()