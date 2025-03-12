import torch
import torch.nn as nn

class TCNEncoder(nn.Module):
    def __init__(self, input_size=1, embedding_size=768, num_channels=[32, 64, 128]):
        super(TCNEncoder, self).__init__()
        
        self.layers = nn.ModuleList()
        in_channels = input_size
        
        # Building TCN layers with dilations (reduced channels for efficiency)
        for out_channels in num_channels:
            self.layers.append(
                nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=2, bias=False)
            )
            self.layers.append(nn.ReLU())
            in_channels = out_channels
        
        # A final convolution layer to bring the output to the desired embedding size (768)
        self.output_layer = nn.Conv1d(in_channels, embedding_size, kernel_size=1)

        # Global Average Pooling to reduce sequence dimension to fixed embedding size
        self.pooling = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x is expected to have shape: [batch_size, sequence_length, num_features]
        x = x.transpose(1, 2)  # Convert to shape [batch_size, num_features, sequence_length]
        
        # Apply TCN layers
        for layer in self.layers:
            x = layer(x)
        
        # Final embedding layer
        x = self.output_layer(x)
        
        # Pooling to get a fixed-size embedding of length 768
        x = self.pooling(x)
        
        # Convert back to shape [batch_size, embedding_size]
        x = x.squeeze(-1)
        
        return x

# Example usage:
# Assuming 'x' is a batch of 1D time series (shape: [batch_size, sequence_length, 1])

input_size = 1  # 1D time series
tcn_encoder = TCNEncoder(input_size=input_size)
tcn_encoder.eval()

# Example time series input (batch size 3, sequence length 100)
x = torch.randn(3, 100, input_size)  # Example: 3 sequences of length 100

with torch.no_grad():
    embeddings = tcn_encoder(x)
print(embeddings.shape)  # Expected: [batch_size, embedding_dim] -> [3, 768]
