from sentence_transformers import SentenceTransformer
import torch

class SentenceBERT(torch.nn.Module):
    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2"):
        super(SentenceBERT, self).__init__()
        self.model = SentenceTransformer(model_name)

    def forward(self, text):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        embeddings = self.model.encode(text, convert_to_tensor=True)
        embeddings = embeddings.to(device)
        return embeddings



def main():
    text_encoder = SentenceBERT()
    text_encoder.eval()

    text_examples = ["Stock price data from 2021 to 2025", "Temperature trend in San Diego"]
    with torch.no_grad():
        embeddings = text_encoder(text_examples)
    print(embeddings.shape)  # Expected: [batch_size, embedding_dim]


if __name__ == "__main__":
    main()
