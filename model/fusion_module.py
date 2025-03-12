import torch
import torch.nn as nn
import torch.nn.init as init

class LinearFusion(nn.Module):
    def __init__(self, input_size_numeric=128, input_size_visual=768, input_size_text=768, output_size=768):
        super(LinearFusion, self).__init__()
        
        # Define the linear layer that combines all embeddings into the final output
        self.linear = nn.Linear(input_size_numeric + input_size_visual + input_size_text, output_size)

        # Apply initialization
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            # Xavier initialization for linear layers
            init.xavier_normal_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)

    def forward(self, numeric_embedding, visual_embedding, text_embedding):
        # Concatenate embeddings from all modalities
        merged_embedding = torch.cat((numeric_embedding, visual_embedding, text_embedding), dim=1)
        
        # Pass through the linear layer
        output = self.linear(merged_embedding)
        
        return output



def main():
    input_size_numeric = 128
    input_size_visual = 768
    input_size_text = 768
    output_size = 768

    fusion_module = LinearFusion(input_size_numeric, input_size_visual, input_size_text, output_size)
    fusion_module.eval()

    # Example embeddings (batch size = 3)
    numeric_embedding = torch.randn(3, input_size_numeric)  # Example: 3 samples, 128 features
    visual_embedding = torch.randn(3, input_size_visual)   # Example: 3 samples, 768 features
    text_embedding = torch.randn(3, input_size_text)       # Example: 3 samples, 768 features

    with torch.no_grad():
        merged_embedding = fusion_module(numeric_embedding, visual_embedding, text_embedding)

    print(merged_embedding.shape)  # Expected output shape: [3, 768]

if __name__ == "__main__":
    main()

