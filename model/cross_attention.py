import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Tokenizer

class CrossAttentionWithPrototypes(nn.Module):
    def __init__(self, prototype_words, fused_emb_size=768, model_name="gpt2"):
        super(CrossAttentionWithPrototypes, self).__init__()

        # Load pre-trained GPT-2 model for word embeddings
        self.llm = GPT2Model.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.prototype_embeddings = self._get_frozen_embeddings(prototype_words)
        self.attn = nn.MultiheadAttention(embed_dim=fused_emb_size, num_heads=8, batch_first=True)

    def _get_frozen_embeddings(self, words):
        """Retrieve frozen word embeddings for prototype words."""
        with torch.no_grad():  # Disable gradient computation
            token_ids = self.tokenizer(words, padding=True, return_tensors="pt",)["input_ids"]
            embeddings = self.llm.wte(token_ids).mean(dim=1)  # take the mean if a prototype is split into multiple subwords

        return embeddings.detach()  # Detach to prevent gradient updates

    def forward(self, fused_embedding):
        """
        Compute cross-attention between the fused multi-modal embedding (Q) 
        and the frozen text prototype embeddings (K, V).
        
        Args:
            fused_embedding: Tensor of shape (batch_size, 1, 768) -> Query (Q)
        
        Returns:
            attended_embedding: Tensor of shape (batch_size, 1, 768)
        """
        # Ensure prototype embeddings are on the same device as input
        device = fused_embedding.device
        K_V = self.prototype_embeddings.to(device)  # (100, 768)

        # Reshape K and V to match MultiheadAttention format (num_prototypes, batch_size, emb_dim)
        K_V = K_V.unsqueeze(1).repeat(1, fused_embedding.shape[0], 1)  # (100, batch_size, 768)
        K_V = K_V.permute(1, 0, 2)  # (batch_size, 100, 768) to match MHA format

        # MultiheadAttention expects (batch_size, seq_len, embed_dim), so we add a sequence dimension to Q
        attended_embedding, _ = self.attn(fused_embedding, K_V, K_V)  # Cross-attention

        return attended_embedding

def main():
    prototype_words = [
    "stabilize", "spike", "increase", "drop", "plateau", "fluctuate", "in the beginning", "sudden rise",
    "peak", "valley", "steady growth", "sharp decline", "gradual increase", "gradual decrease",
    "oscillate", "volatile", "steady", "outlier", "anomaly", "trend", "seasonal", "cyclical",
    "mean-reverting", "acceleration", "deceleration", "surge", "dip", "persistent", "transient",
    "burst", "regression", "correlation", "autoregressive", "dampening", "converge", "diverge",
    "periodic", "non-stationary", "stationary", "residual", "cumulative", "saturation",
    "uptrend", "downtrend", "breakout", "shock", "rebound", "flatten", "persist", "retrace",
    "noise", "random walk", "drift", "inflection point", "moving average", "momentum",
    "overshoot", "undershoot", "lagging", "leading", "slope", "spurt", "compression",
    "expansion", "volatility cluster", "reversal", "lag", "contraction", "extension",
    "signal", "mean shift", "jump", "long-term", "short-term", "irregular", "knee point",
    "asymptote", "equilibrium", "breaking point", "threshold", "compression", "dilation",
    "decay", "growth rate", "exponential rise", "logarithmic growth", "relative change",
    "absolute change", "tipping point", "structural break", "convex", "concave",
    "random fluctuation", "cyclicality", "persistent oscillation", "transient spike"
]  # hand-picked prototypes

    cross_attn = CrossAttentionWithPrototypes(prototype_words)
    cross_attn.eval()

    fused_embedding = torch.randn(3, 1, 768)  # Example batch of 3 samples
    output_embedding = cross_attn(fused_embedding)

    print(output_embedding.shape)  # Expected: (batch size, seq len, emb size)


if __name__ == "__main__":
    main()
