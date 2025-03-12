from text_encoder import SentenceBERT
from visual_encoder import ViTEncoder
from ts_encoder import TCNEncoder
from fusion_module import LinearFusion
import torch
import torch.nn as nn


class Mobtep(torch.nn.Module):
    def __init__(self, tcn_emb_size=128):
        super(Mobtep, self).__init__()
        
        # Pre-trained text and visual encoders, and TCN encoder
        self.text_encoder = SentenceBERT()  # Pretrained
        self.visual_encoder = ViTEncoder()  # Pretrained
        self.ts_encoder = TCNEncoder(embedding_size=tcn_emb_size)  # From scratch
        self.text_prototypes = None # the unified embedding does dot product with text prototypes
        self.caption_generator = None # the top active text prototypes are fed to the caption_generator to generate captions
        
        self.fusion_module = LinearFusion(input_size_numeric=tcn_emb_size, input_size_visual=768, input_size_text=768, output_size=768)

    def forward(self, ts_input, text_input, visual_input):
        # Pass the time series data through the TCN encoder
        ts_embedding = self.ts_encoder(ts_input)
        
        # Pass the text data through the text encoder
        text_embedding = self.text_encoder(text_input)
        
        # Pass the visual data through the visual encoder
        visual_embedding = self.visual_encoder(visual_input)
        
        # Merge the embeddings from all modalities using the fusion module
        fused_embedding = self.fusion_module(ts_embedding, visual_embedding, text_embedding)
        return fused_embedding



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mobtep = Mobtep(tcn_emb_size=128).to(device)
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


