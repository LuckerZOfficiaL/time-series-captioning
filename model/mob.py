from helpers import(
    load_config
)
from internVL import(
    batch_inference
)
from chronos_embedder import ChronosEmbedder
import torch
from transformers import AutoTokenizer, AutoModel
import os

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
        
        ts_emb = self.chronos(ts, pooling=pooling)  # This is currently unused, so mob is really just internVL
        responses = batch_inference(self.internvl, self.internvl_tokenizer, image_paths, prompts, max_output_tokens=max_output_tokens)
        
        return ts_emb, responses

        
def evaluate_mob(model, ts_folder_path, metadata_folder_path, image_folder_path, save_folder_path, batch_size=32):
    config = load_config()
    ts_list = []
    prompt_list = []
    image_paths = []

    ts_files = os.listdir(ts_folder_path)
    ts_files = [filename for filename in ts_files if filename not in os.listdir(save_folder_path)] # skip the captions that are already generated
    ts_files.sort()
    print(f"\n{len(ts_files)} captions yet to be generated.")
    
    batch_start_idx = 0

    while batch_start_idx < len(ts_files):
        batch_end_idx = min(len(ts_files), batch_start_idx + batch_size)
        print(f"\nPreparing inputs for batch {batch_start_idx}-{batch_end_idx-1}...")
        for i in range(batch_start_idx, batch_end_idx):
            filename = ts_files[i]
            #print("Preparing inputs for: ", filename[:-4])

            # Read ts and append it to a list, later to be tensorized after the loop
            with open(ts_folder_path+"/"+filename, 'r') as file:
                lines = file.read().splitlines()        
            for line in lines:
                # Convert each value in the line to a float
                values = [float(value) for value in line.split()]
                # Convert the list of values into a tensor
                tensor = torch.tensor(values)
                ts_list.append(tensor)

            # Read metadata, prepend a prompt to it and add to the prompt list
            metadata_filename = metadata_folder_path+"/"+filename[:-4]+".json"
            with open(metadata_filename, 'r') as metadata_file:
                metadata_text = metadata_file.read()
            prompt = f"""Provide a description for the following line plot, given these auxiliary metadata: 
            \n
            {metadata_text}
            \n
            Give your answer in a single paragraph, without additional explanations or formatting.        
            """
            prompt_list.append(prompt)
            
            # Append image path
            image_paths.append(image_folder_path+"/"+filename[:-4]+".jpeg")

        stacked_ts = torch.stack(ts_list, dim=0)

        print(f"\nGenerating captions for batch {batch_start_idx}-{batch_end_idx-1}...")
        _, responses = model(prompt_list, image_paths, stacked_ts, 
                            pooling=config['mobtep']['chronos_pooling'], 
                            max_output_tokens=config['mobtep']['max_output_tokens'])
        
        for i in range(batch_start_idx, batch_end_idx):
            response_file = f"{save_folder_path+"/"+ts_files[i]}"
            with open(response_file, 'w') as file:
                file.write(responses[i])
        
        batch_start_idx = batch_end_idx

def main():
    # Rest of the code
    config = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    mob = Mob(chronos_name=config['mobtep']['chronos_name'], internvl_name=config['mobtep']['internvl_name']).to(device)
    
    """ts = torch.randn(2, 20, 1)
    image_paths = ['/home/ubuntu/thesis/data/samples/plots/air quality_0.jpeg',
                '/home/ubuntu/thesis/data/samples/plots/demography_0.jpeg']

    prompts = ['Describe this line chart about the hourly CO levels in London. Discuss the values you see.', 
                'Describe this line chart about the yearly death rates in Greece. Discuss the values you see.']

    ts_embed, responses = mob(prompts, image_paths, ts, 
                                pooling=config['mobtep']['chronos_pooling'], 
                                max_output_tokens=config['mobtep']['max_output_tokens'])

    print(f"\nts_embed: {ts_embed.shape}\n\nresponses:\n")
    for response in responses:
        print("\n", response)"""
    
    ts_folder_path = "/home/ubuntu/thesis/data/samples/time series"
    metadata_folder_pth = "/home/ubuntu/thesis/data/samples/metadata"
    image_folder_path = "/home/ubuntu/thesis/data/samples/plots"
    save_folder_path="/home/ubuntu/thesis/data/samples/captions/mob"
    evaluate_mob(mob, ts_folder_path, metadata_folder_pth, image_folder_path, save_folder_path, batch_size=100)




# You might neet to run this script many times without any change since there are too many examples to fit into memory for a single run. 
if __name__ == "__main__":
    main()
    