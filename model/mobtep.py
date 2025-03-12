from text_encoder import SentenceBERT
from visual_encoder import ViTEncoder
from ts_encoder import TCNEncoder
from fusion_module import LinearFusion
from cross_attention import CrossAttentionWithPrototypes
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import torch.nn as nn


class Mobtep(torch.nn.Module):
    def __init__(self, prototype_words, tcn_emb_size=64):
        super(Mobtep, self).__init__()
        
        # Pre-trained text and visual encoders, and TCN encoder
        self.text_encoder = SentenceBERT()  # Pretrained
        self.visual_encoder = ViTEncoder()  # Pretrained
        self.ts_encoder = TCNEncoder(embedding_size=tcn_emb_size)  # From scratch

        self.fusion_module = LinearFusion(input_size_numeric=tcn_emb_size, input_size_visual=768, input_size_text=768, output_size=768)
        self.prototype_attention = CrossAttentionWithPrototypes(prototype_words) # the unified embedding does cross-attention with text prototypes
        #self.linear_proj = nn.Linear(768, 768)

        self.caption_generator = GPT2LMHeadModel.from_pretrained("gpt2")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        #nn.init.xavier_uniform_(self.linear_proj.weight)
        #nn.init.zeros_(self.linear_proj.bias)
        

    def forward(self, ts_input, text_input, visual_input, generate_text=False):
        # Encoding each modality
        numeric_embedding = self.ts_encoder(ts_input)
        text_embedding = self.text_encoder(text_input)
        visual_embedding = self.visual_encoder(visual_input)

        # Fusion of embeddings (not shown in this snippet)
        fused_embedding = self.fusion_module(numeric_embedding, visual_embedding, text_embedding)

        # Cross-attention with text prototypes
        attended_embedding = self.cross_attention(fused_embedding)

        # Linear projection to match GPT-2's embedding space
        projected_embedding = self.linear_proj(attended_embedding)

        if generate_text:
            # Generate description using GPT-2's language model
            captions = self.generate_captions(projected_embedding)
            return captions

        return projected_embedding

    def generate_captions(self, aligned_embedding):
        """
        Use GPT-2 to generate a description based on the aligned embedding.
        
        Args:
            aligned_embedding: Tensor of shape (batch_size, 1, embedding_size)
        
        Returns:
            Generated description as a list of strings for the batch.
        """
        # Convert the aligned embedding to the correct format for GPT-2 input
        input_ids = aligned_embedding.squeeze(1)  # Remove the sequence dimension (shape: batch_size, embedding_size)

        # Ensure the input tensor has the correct shape (batch_size, sequence_length)
        input_ids = input_ids.unsqueeze(1)  # (batch_size, 1, embedding_size) -> (batch_size, 1, 768)

        # Use GPT-2's generation API to create a description
        outputs = self.gpt2.generate(
            input_ids=input_ids,  # Embedding context for generation
            max_length=50,  # Max length of generated description
            num_beams=5,  # Beam search for better quality text
            no_repeat_ngram_size=2,  # Avoid repetition
            early_stopping=True,
            num_return_sequences=1,  # Number of output sequences to return
        )

        # Decode the output token IDs to text (batch processing)
        generated_descriptions = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        return generated_descriptions



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
]


    mobtep = Mobtep(tcn_emb_size=128, prototype_words=prototype_words).to(device)
    mobtep.eval()

    # Example input data (random for demonstration)
    ts_input = torch.randn(3, 100, 1).to(device)  # Example time series (3 samples, length 100)
    text_input = ["This is a description of the time series.", "Another description here.", "More descriptions."]
    visual_input = torch.randn(3, 3, 224, 224).to(device)  # Example visual data (3 samples, H224 x W224 images)

    # Forward pass
    with torch.no_grad():
        output = mobtep(ts_input, text_input, visual_input)

    print(output.shape)  # Expected: [3, 768]


if __name__ == "__main__":
    main()


