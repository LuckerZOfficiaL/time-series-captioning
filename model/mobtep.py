from text_encoder import SentenceBERT
from visual_encoder import ViTEncoder
from ts_encoder import TCNEncoder
from fusion_module import LinearFusion
from cross_attention import CrossAttentionWithPrototypes
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_cosine_similarity(embeddings):
            # Reshape the input embeddings to [batch_size, embedding_size]
            embeddings = embeddings.view(embeddings.size(0), -1)
            
            # Compute the dot product of the embeddings
            dot_product = torch.matmul(embeddings, embeddings.t())
            
            # Compute the norm of the embeddings
            norm = torch.norm(embeddings, dim=1, keepdim=True)
            
            # Compute the cosine similarity matrix
            cosine_similarity = dot_product / (norm * norm.t())
            
            return cosine_similarity


class Mobtep(torch.nn.Module):
    def __init__(self, prototype_words, tcn_emb_size=64, use_linear_proj=False):
        super(Mobtep, self).__init__()
        
        # Pre-trained text and visual encoders, and TCN encoder
        self.text_encoder = SentenceBERT()  # Pretrained
        self.visual_encoder = ViTEncoder()  # Pretrained
        self.ts_encoder = TCNEncoder(embedding_size=tcn_emb_size)  # From scratch

        self.fusion_module = LinearFusion(input_size_numeric=tcn_emb_size, input_size_visual=768, input_size_text=768, output_size=768)
        self.prototype_attention = CrossAttentionWithPrototypes(prototype_words) # the unified embedding does cross-attention with text prototypes
        
        self.use_linear_proj = False
        if use_linear_proj:
            self.use_linear_proj = True
            self.linear_proj = nn.Linear(768, 768)
            nn.init.xavier_uniform_(self.linear_proj.weight)
            nn.init.zeros_(self.linear_proj.bias)

        self.caption_generator = GPT2LMHeadModel.from_pretrained("gpt2")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        
        

    def forward(self, ts_input, text_input, visual_input, output_text=False):
        # Encoding each modality
        numeric_embedding = self.ts_encoder(ts_input)
        #print("Numeric embedding: ", numeric_embedding.shape)
        text_embedding = self.text_encoder(text_input)
        #print("Text embedding: ", text_embedding.shape)
        visual_embedding = self.visual_encoder(visual_input)
        #print("Visual embedding: ",visual_embedding.shape)

        # Fusion of embeddings (not shown in this snippet)
        fused_embedding = self.fusion_module(numeric_embedding, visual_embedding, text_embedding)
        #print("Fused: ", fused_embedding.shape)
        
        fused_embedding = fused_embedding.unsqueeze(1)  # Add a new dimension
        #print("Fused and unsqueezed: ", fused_embedding.shape)
        #print(compute_cosine_similarity(fused_embedding))
        

        # Cross-attention with text prototypes
        x = self.prototype_attention(fused_embedding)
        #print("Prototyped: ", x.shape)

        if self.use_linear_proj:
            # Linear projection between embedding and caption generation
            x = self.linear_proj(x)
            #print("Projected: ", x.shape)

        #print(compute_cosine_similarity(x))

        if output_text:
            # Generate description using GPT-2's language model
            captions = self.generate_captions(x)
            return captions

        return x

    def generate_captions(self, aligned_embedding):
        """
        Use GPT-2 to generate a description based on the aligned embedding.
        
        Args:
            aligned_embedding: Tensor of shape (batch_size, 1, embedding_size)
        
        Returns:
            Generated description as a list of strings for the batch.
        """
        batch_size = aligned_embedding.shape[0]
        
        # Ensure the input tensor has the correct shape (batch_size, sequence_length, embedding_size)
        inputs_embeds = aligned_embedding  # Already has shape (batch_size, 1, 768)

        # Create an attention mask (all ones, since we have a single valid token)
        attention_mask = torch.ones((batch_size, 1), dtype=torch.long, device=aligned_embedding.device)

        # Use GPT-2's generation API to create a description
        outputs = self.caption_generator.generate(
            inputs_embeds=inputs_embeds,  # Use embeddings instead of token IDs
            attention_mask=attention_mask,  # Explicitly pass attention mask
            max_new_tokens=300,  # Generate up to 300 new tokens
            num_beams=5,  # Beam search for better quality text
            no_repeat_ngram_size=2,  # Avoid repetition
            early_stopping=True,
            num_return_sequences=1,  # Number of output sequences to return
            pad_token_id=self.tokenizer.eos_token_id,  # Ensure padding behavior is correct
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


    mobtep = Mobtep(tcn_emb_size=128, prototype_words=prototype_words, use_linear_proj=False).to(device)
    mobtep.eval()

    # Example input data (random for demonstration)
    ts_input = torch.randn(3, 100, 1).to(device)  # Example time series (3 samples, length 100)
    text_input = ["Tell me a story.", "How are you?", "I am Luca."]
    visual_input = torch.randn(3, 3, 224, 224).to(device)  # Example visual data (3 samples, 3 channels, H224 x W224 images)

    # Forward pass
    with torch.no_grad():
        output = mobtep(ts_input, text_input, visual_input, output_text=True)

    #print(output.shape)  # Expected shape: [3, 1, 768]
    for i, caption in enumerate(output):
        print("\n\ni)\n", caption)


if __name__ == "__main__":
    main()


