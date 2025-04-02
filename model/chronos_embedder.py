import torch
from chronos import ChronosPipeline
import numpy as np
from helpers import(
    load_config
)


class ChronosEmbedder(torch.nn.Module):
    def __init__(self, model_name="amazon/chronos-t5-small"):
        super(ChronosEmbedder, self).__init__()
        self.chronos = ChronosPipeline.from_pretrained(model_name, )
    
    def mean_pooling(self, embeddings):
        return torch.mean(embeddings, dim=1) 
        
    def max_pooling(self, embeddings):
        return torch.max(embeddings, dim=1).values

    def forward(self, time_series, pooling="mean"):
        time_series = time_series.squeeze(-1)
        embeddings, tokenizer_state = self.chronos.embed(time_series)
        #print(embeddings.shape)
        if pooling == "mean":
            embeddings = self.mean_pooling(embeddings)
        elif pooling == "max":
            embeddings = self.max_pooling(embeddings)
        return embeddings



def main():
    config = load_config()
    embedder = ChronosEmbedder(model_name = config['mobtep']['chronos_name'])

    time_series = torch.randn(4, 20, 1)

    embeddings = embedder(time_series, pooling=config['mobtep']['chronos_pooling'])

    print("Embeddings shape:", embeddings.shape)  # (batch_size, context_length, embedding_dim)


if __name__ == "__main__":
    main()