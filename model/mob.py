from helpers import(
    load_config
)
from internVL import(
    batch_inference
)
from chronos_embedder import ChronosEmbedder
import torch
from transformers import AutoTokenizer, AutoModel

class Mob(torch.nn.Module):
    def __init__(self, chronos_name="amazon/chronos-t5-small", internvl_name="OpenGVLab/InternVL2_5-2B"):
        super(Mob, self).__init__()
        self.chronos = ChronosEmbedder(model_name=chronos_name)
        self.internvl =  AutoModel.from_pretrained(
                                                    internvl_name,
                                                    torch_dtype=torch.bfloat16,
                                                    #low_cpu_mem_usage=True,
                                                    #use_flash_attn=True,
                                                    trust_remote_code=True)
        self.internvl_tokenizer = AutoTokenizer.from_pretrained(internvl_name, trust_remote_code=True, use_fast=False)

    def forward(self, prompts, image_paths, ts, pooling="mean", max_output_tokens=250): 
        """
            args:
                prompts: a list of B text prompts
                image_paths: a list of B images filepaths
                ts: a tensor of shape (B, seq_len, 1)
        """
        
        ts_emb = self.chronos(ts, pooling=pooling)  # Chronos embedder should handle device internally
        responses = batch_inference(self.internvl, self.internvl_tokenizer, image_paths, prompts, max_output_tokens=max_output_tokens)
        
        return ts_emb, responses

        


def main():
    config = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    mob = Mob(chronos_name=config['mobtep']['chronos_name'], internvl_name=config['mobtep']['internvl_name']).to(device)
    
    ts = torch.randn(2, 20, 1)
    image_paths = ['/home/ubuntu/thesis/data/samples/plots/air quality_0.jpeg',
                '/home/ubuntu/thesis/data/samples/plots/demography_0.jpeg']

    prompts = ['Describe this line chart about the hourly CO levels in London. Discuss the values you see.', 
                'Describe this line chart about the yearly death rates in Greece. Discuss the values you see.']

    ts_embed, responses = mob(prompts, image_paths, ts, 
                                pooling=config['mobtep']['chronos_pooling'], 
                                max_output_tokens=config['mobtep']['max_output_tokens'])

    print(f"\nts_embed: {ts_embed.shape}\n\nresponses:\n")
    for response in responses:
        print("\n", response)

if __name__ == "__main__":
    main()