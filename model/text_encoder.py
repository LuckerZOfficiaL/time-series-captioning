from sentence_transformers import SentenceTransformer
import torch

class TextEncoder(torch.nn.Module):
    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2"):
        super(TextEncoder, self).__init__()
        self.model = SentenceTransformer(model_name)

    def forward(self, text):
        return self.model.encode(text, convert_to_tensor=True)  # Output: (embedding_dim,)

# Example usage
text_encoder = TextEncoder()
text_encoder.eval()

text_examples = ["Stock price data from 2021 to 2025", "Temperature trend in San Diego"]
with torch.no_grad():
    embeddings = text_encoder(text_examples)
print(embeddings.shape)  # Expected: [batch_size, embedding_dim]