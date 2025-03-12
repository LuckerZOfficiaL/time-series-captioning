import torch
import torch.nn as nn
import torch.nn.init as init

class TCNEncoder(nn.Module):
    def __init__(self, input_size=1, embedding_size=128, num_channels=[32, 64, 128]):
        super(TCNEncoder, self).__init__()
        
        self.layers = nn.ModuleList()
        in_channels = input_size
        
        # Building TCN layers with dilations (reduced channels for efficiency)
        for out_channels in num_channels:
            conv_layer = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=2, bias=True)
            self.layers.append(conv_layer)
            self.layers.append(nn.ReLU())
            in_channels = out_channels
        
        # A final convolution layer to bring the output to the desired embedding size (128)
        self.output_layer = nn.Conv1d(in_channels, embedding_size, kernel_size=1)

        # Global Average Pooling to reduce sequence dimension to fixed embedding size
        self.pooling = nn.AdaptiveAvgPool1d(1)
        
        # Apply weight initialization
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            # He initialization for convolutional layers
            init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            # Xavier initialization for linear layers
            init.xavier_normal_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)
        # ReLU layers do not need initialization, so we skip them

    def forward(self, x):
        # x is expected to have shape: [batch_size, sequence_length, num_features]
        x = x.transpose(1, 2)  # Convert to shape [batch_size, num_features, sequence_length]
        
        # Apply TCN layers
        for layer in self.layers:
            x = layer(x)
        
        # Final embedding layer
        x = self.output_layer(x)
        
        # Pooling to get a fixed-size embedding of length 128 (or the desired output size)
        x = self.pooling(x)
        
        # Convert back to shape [batch_size, embedding_size]
        x = x.squeeze(-1)
        
        return x



def main():
    input_size = 1  # 1D time series
    tcn_encoder = TCNEncoder(input_size=input_size)
    tcn_encoder.eval()

    # Example time series input (batch size 3, sequence length 100)
    x = torch.randn(3, 100, input_size)  # Example: 3 sequences of length 100

    with torch.no_grad():
        embeddings = tcn_encoder(x)
    print(embeddings.shape)  # Expected: [batch_size, embedding_dim] -> [3, 128]

if __name__ == "__main__":
    main()

