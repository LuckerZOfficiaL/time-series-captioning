from helpers import(
    load_config,
    generate_prompt_for_baseline
)
from internVL import(
    batch_inference,
    mob_batch_inference
)
from train_internVL import(
    InternVLDataset,
    create_dataloaders,
    compute_loss,
)
from custom_methods import(
    custom_forward,
    custom_generate,
    custom_batch_chat
)
from torch.optim import AdamW
from chronos_embedder import ChronosEmbedder
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import os
from tqdm import tqdm
from typing import Optional
import types
import json



class Mob(nn.Module):
    def __init__(self, chronos_name="amazon/chronos-t5-small", internvl_name="OpenGVLab/InternVL2_5-2B", projector_init="zero", sum_ts_emb_to="all"):
        super(Mob, self).__init__()
        self.chronos = ChronosEmbedder(model_name=chronos_name)
        self.projector = nn.Linear(512, 2048) # 512 and 2048 are the embedding sizes of Chronos and InternVL
    
        self.gate_proj = nn.Linear(2048, 1)
        self.sigmoid = nn.Sigmoid() # the sigmoid outputs a scaling factor between 0 and 1 used to sum the chronos embedding to the internVL input embedding
        
        self.internvl = AutoModel.from_pretrained(
                                internvl_name,
                                torch_dtype=torch.bfloat16,
                                #low_cpu_mem_usage=True,
                                use_flash_attn=True,
                                trust_remote_code=True)
        self.internvl_tokenizer = AutoTokenizer.from_pretrained(internvl_name, trust_remote_code=True, use_fast=False)
        self.sum_ts_emb_to = sum_ts_emb_to
        
        
        for param in self.chronos.chronos.model.parameters():
            param.requires_grad = False
        #for param in self.internvl.parameters():
        #    param.requires_grad = False
            
        if projector_init == "zero":
            nn.init.zeros_(self.projector.weight)
            nn.init.zeros_(self.gate_proj.weight)
        elif projector_init == "almost zero":
            nn.init.normal_(self.projector.weight, mean=0.0, std=1e-4)
            nn.init.normal_(self.gate_proj.weight, mean=0.0, std=1e-4)
        elif projector_init == "xavier":
            nn.init.xavier_uniform_(self.projector.weight)
            nn.init.xavier_uniform_(self.gate_proj.weight)
        else:
            pass # the default initialization is Kaiming uniform
        
        # replace the original methods with my custom ones, which accomodate ts embedding injection
        self.internvl.generate = types.MethodType(custom_generate, self.internvl)
        self.internvl.forward = types.MethodType(custom_forward, self.internvl)
        self.internvl.batch_chat = types.MethodType(custom_batch_chat, self.internvl)

    def forward(self, ts, pixel_values, input_ids, attention_mask, target_ids, pooling="mean",use_chronos=True):
        if ts is not None and use_chronos:  # if ts tensor is None (not provided), just skip chronos and use internVL directly
            ts_emb = self.chronos(ts, pooling=pooling).to("cuda")
            ts_emb = self.projector(ts_emb) 
            ts_emb = ts_emb
            
            coef = self.sigmoid(self.gate_proj(ts_emb)) # sigmoid to learn a dynamic scaling coefficient for summing the chronos ts embedding to the input embeddings of internVL
            ts_emb =  (coef * ts_emb).to(torch.bfloat16)
            #print(f"First Coef: {coef[0]}")
        else: ts_emb = None
            
        pixel_values, input_ids, attention_mask = pixel_values.to("cuda"), input_ids.to("cuda"), attention_mask.to("cuda")
        
        if target_ids is not None:
            target_ids = target_ids.to("cuda")
            
        internvl_outputs = self.internvl(ts_emb = ts_emb,
                                        sum_ts_emb_to=self.sum_ts_emb_to,
                                        pixel_values=pixel_values, 
                                        input_ids=input_ids,
                                        labels=target_ids,
                                        image_flags=torch.ones(input_ids.size(0)),
                                        attention_mask=attention_mask)
        outputs = internvl_outputs
        return outputs # currnetly, ts_emb from chronos is unused


    def batch_chat(self, ts, pixel_values, questions, generation_config, pooling="mean", num_patches_list=None, use_chronos=True, history=None, return_history=False, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>',
    IMG_CONTEXT_TOKEN='<IMG_CONTEXT>', verbose=False, image_counts=None):
        if ts is not None and use_chronos:
            ts_emb = self.chronos(ts, pooling=pooling).to("cuda")
            ts_emb = self.projector(ts_emb) 
            ts_emb = ts_emb
            
            coef = self.sigmoid(self.gate_proj(ts_emb)) # sigmoid to learn a dynamic scaling coefficient for summing the chronos ts embedding to the input embeddings of internVL
            ts_emb =  (coef * ts_emb).to(torch.bfloat16)
            
        else: ts_emb=None
        
        return self.internvl.batch_chat(tokenizer=self.internvl_tokenizer, 
                                        ts_emb=ts_emb,
                                        sum_ts_emb_to=self.sum_ts_emb_to ,
                                        pixel_values=pixel_values, 
                                        questions=questions, generation_config=generation_config,
                                        num_patches_list=num_patches_list,
                                        history=history, 
                                        return_history=False, 
                                        IMG_START_TOKEN='<img>', 
                                        IMG_END_TOKEN='</img>',
                                        IMG_CONTEXT_TOKEN='<IMG_CONTEXT>', 
                                        verbose=False, image_counts=None)
    

    def get_responses(self, prompts, image_paths, ts, pooling="mean", max_output_tokens=256, use_chronos=True): 
        """
            args:
                prompts: a list of B text prompts
                image_paths: a list of B images filepaths
                ts: a tensor of shape (B, seq_len, 1)
                use_chronos: if chronos ts embedding is used, if False, it's equivalent to running internVL only
        """            
        responses = mob_batch_inference(model=self, image_paths=image_paths, prompts=prompts, ts=ts, max_output_tokens=max_output_tokens)
        return responses
    
    
    
def generate_captions(model, ts_folder_path, metadata_folder_path, image_folder_path, save_folder_path, batch_size=32, use_chronos=False):
    config = load_config()
    model.eval()
    ts_files = os.listdir(ts_folder_path)
    ts_files = [filename for filename in ts_files if filename not in os.listdir(save_folder_path)] # skip the captions that are already generated
    ts_files.sort()
    print(f"\n{len(ts_files)} captions yet to be generated.")
    
    batch_start_idx = 0

    while batch_start_idx < len(ts_files):
        ts_list = []
        prompt_list = []
        image_paths = []
    
        max_ts_len=0
        batch_end_idx = min(len(ts_files), batch_start_idx + batch_size)
        print(f"\nPreparing inputs for batch {batch_start_idx}-{batch_end_idx-1}...")
        for i in range(batch_start_idx, batch_end_idx):
            filename = ts_files[i]
            dataset_name = filename.split("_")[0]
            #print("Preparing inputs for: ", filename[:-4])

            # Read ts and append it to a list, later to be tensorized after the loop
            with open(ts_folder_path+"/"+filename, 'r') as file:
                lines = file.read().splitlines()        
                
            values = [float(value) for value in lines]
            #print("\nValues: ", values)
            max_ts_len = max(len(values), max_ts_len)
            # Convert the list of values into a tensor
            tensor = torch.tensor(values)
            ts_list.append(tensor)

            # Read metadata, generate a prompt out of it and append to the prompt list
            metadata_filename = metadata_folder_path+"/"+filename[:-4]+".json"
            
            with open(metadata_filename, 'r') as metadata_file:
                metadata = json.load(metadata_file)
                
                
            prompt = generate_prompt_for_baseline(dataset_name=dataset_name, metadata=metadata, ts=values)
            prompt = prompt + "\nI have attached the line plot of the time series to help you."
            prompt_list.append(prompt)
                        
            # Append image path
            image_paths.append(image_folder_path+"/"+filename[:-4]+".jpeg")

        for i in range(len(ts_list)): # apply left-padding 
            if ts_list[i].size(0) < max_ts_len:
                padding = torch.full((max_ts_len - ts_list[i].size(0),), torch.nan)
                ts_list[i] = torch.cat((padding, ts_list[i]), dim=0)
                #print("ts: ", ts_list[i].shape)
        
        #print("max ts len ", max_ts_len)
        #print("ts shape ", ts_list[0].shape)
        stacked_ts = torch.stack(ts_list, dim=0)
        
        #print("stacked shape: ", stacked_ts.shape)

        print(f"Generating captions for batch {batch_start_idx}-{batch_end_idx-1}...")
        if use_chronos:
            responses = model.get_responses(prompt_list, image_paths, ts=stacked_ts, sum_ts_emb_to=model.sum_ts_emb_to,
                            pooling=config['mobtep']['chronos_pooling'], 
                            max_output_tokens=config['mobtep']['max_output_tokens'])
        else:
            responses = model.get_responses(prompt_list, image_paths, ts=None, 
                            pooling=config['mobtep']['chronos_pooling'], 
                            max_output_tokens=config['mobtep']['max_output_tokens'])
        
        for i in range(batch_start_idx, batch_end_idx):
            response_file = f"{save_folder_path+"/"+ts_files[i]}"
            with open(response_file, 'w') as file:
                file.write(responses[i-batch_start_idx])
        
        batch_start_idx = batch_end_idx


def train_mob(model, train_loader, optimizer, ignore_id, epochs=5, val_loader=None): # if val_loader is not None, it also does validation
    print("\nStart Training...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config=load_config()
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # TRAINING
        total_train_loss = 0
        model.train()
        for ts, pixel_values, input_ids, attention_mask, target_ids, target_attention_mask in tqdm(train_loader):
            pixel_values = pixel_values.cuda().to(torch.bfloat16)
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
            target_ids = target_ids.cuda()
            #target_attention_mask = target_attention_mask.cuda() # useless

            outputs = model(ts=ts, 
                            pixel_values=pixel_values, 
                            input_ids=input_ids,
                            target_ids=None,
                            attention_mask=attention_mask,
                            pooling=config['mobtep']['chronos_pooling'],
                            use_chronos=config['mobtep']['use_chronos'])
            #print(f"logits shape {outputs.logits.shape}, target shape {target_ids.shape}")
            loss = compute_loss(outputs.logits, target_ids, ignore_id=ignore_id)
            #loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {total_train_loss/len(train_loader):.4f}")
        train_losses.append(total_train_loss/len(train_loader))
        
        # VALIDATION
        if val_loader is not None:
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for ts, val_pixel_values, val_input_ids, val_attention_mask, val_target_ids, val_target_attention_mask in tqdm(val_loader):
                    val_pixel_values = val_pixel_values.cuda().to(torch.bfloat16)
                    val_input_ids = val_input_ids.cuda()
                    val_attention_mask = val_attention_mask.cuda()
                    val_target_ids = val_target_ids.cuda()
                    val_outputs = model(ts=ts,
                                        pixel_values=val_pixel_values, 
                                        input_ids=val_input_ids,
                                        target_ids=val_target_ids,
                                        attention_mask=val_attention_mask,
                                        use_chronos=config['mobtep']['use_chronos'])
                    val_loss = compute_loss(val_outputs.logits, val_target_ids, ignore_id=ignore_id)
                    total_val_loss += val_loss.item()
            print(f"Epoch {epoch+1}/{epochs} - Validation Loss: {total_val_loss/len(val_loader):.4f}")
            val_losses.append(total_val_loss/len(val_loader))

    return train_losses, val_losses


def main():
    # Rest of the code
    config = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = Mob(chronos_name=config['mobtep']['chronos_name'], 
                internvl_name=config['mobtep']['internvl_name'],
                projector_init=config['mobtep']['projector_init'],
                sum_ts_emb_to=config['mobtep']['sum_ts_emb_to']).to(device)

    
    train_loader, val_loader = create_dataloaders(model.internvl_tokenizer, 
                                                    ts_folder=config['path']['ts_folder_path'], 
                                                    img_folder=config['path']['plot_folder_path'], 
                                                    metadata_folder=config['path']['metadata_folder_path'], 
                                                    gt_folder=config['path']['gt_captions_folder_path'],
                                                    max_tokens=config['mobtep']['max_output_tokens'],
                                                    train_batch_size=config['train']['batch_size'],
                                                    val_batch_size=config['eval']['batch_size'],
                                                    val_split=config['eval']['val_split'],
                                                    seed=config['general']['random_seed'])

    optimizer = AdamW(model.parameters(), lr=float(config['train']['lr']), weight_decay=float(config['train']['weight_decay']))

    train_losses, val_losses = train_mob(model, 
                            train_loader=train_loader, 
                            optimizer=optimizer, 
                            ignore_id=model.internvl_tokenizer.pad_token_id, 
                            epochs=config['train']['epochs'],
                            val_loader=val_loader)
    print("\nTrain Losses: ", train_losses)
    print("\nVal Losses: ", val_losses)

    
    
    ######################################## SAVING CHECKPOINT #######################################
    filepath = f"{config['path']['checkpoints_folder_path']}/Mob2_5-2B_{round(val_losses[-1], 3) if val_losses != [] else ""}_{config['train']['epochs']}eps.pth"
    torch.save(model.state_dict(), filepath)


    ####################################### TOY DEMO #######################################
    
    #checkpoint_path = "/home/ubuntu/thesis/model/checkpoints/Mob2_5-2B_5.099.pth"
    #model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    #ts = torch.randn(2, 20, 1)
    image_paths = ['/home/ubuntu/thesis/data/samples/new samples no overlap/test/plots/agriculture_0_test.jpeg',
                '/home/ubuntu/thesis/data/samples/new samples no overlap/test/plots/agriculture_1_test.jpeg']

    metadata_paths = ["/home/ubuntu/thesis/data/samples/new samples no overlap/test/metadata/agriculture_0_test.json",
                      "/home/ubuntu/thesis/data/samples/new samples no overlap/test/metadata/agriculture_1_test.json"]
    
    metadata = []
    for metadata_path in metadata_paths: 
        with open(metadata_path, 'r') as metadata_file:
            metadata.append(json.load(metadata_file))
            
    ts_file_paths = ["/home/ubuntu/thesis/data/samples/new samples no overlap/test/time series/agriculture_0_test.txt",
                     "/home/ubuntu/thesis/data/samples/new samples no overlap/test/time series/agriculture_1_test.txt"]
    ts = []
    for ts_file_path in ts_file_paths:
        with open(ts_file_path, 'r') as file:
            lines = file.read().splitlines()
        ts_values = [float(value) for value in lines]
        ts.append(torch.tensor(ts_values).unsqueeze(1)) 

    prompts = [generate_prompt_for_baseline(dataset_name="agriculture", metadata=metadata[0], ts=ts[0]),
               generate_prompt_for_baseline(dataset_name="agriculture", metadata=metadata[1], ts=ts[1])]

    responses = mob_batch_inference(model=model,ts=None, image_paths=image_paths, prompts=prompts, max_output_tokens=256)

    print(f"\nResponses:\n")
    for response in responses:
        print("\n---------------------------------------------------------------------------------------\n", response)

# You might neet to run this script many times without any change since there are too many examples to fit into memory for a single run. 
if __name__ == "__main__":
    #main()
    
    ################### GENERATE CAPTION FILES ##################################
    config = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    model = Mob(chronos_name=config['mobtep']['chronos_name'], 
                internvl_name=config['mobtep']['internvl_name'],
                projector_init=config['mobtep']['projector_init'],
                sum_ts_emb_to=config['mobtep']['sum_ts_emb_to']).to(device)
    
    checkpoint_path = f"/home/ubuntu/thesis/model/checkpoints/{config['mobtep']['mob_checkpoint']}.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    ts_folder_path = "/home/ubuntu/thesis/data/samples/new samples no overlap/test/time series"
    metadata_folder_pth = "/home/ubuntu/thesis/data/samples/new samples no overlap/test/metadata"
    image_folder_path = "/home/ubuntu/thesis/data/samples/new samples no overlap/test/plots"
    save_folder_path= f"/home/ubuntu/thesis/data/samples/new samples no overlap/generated captions/internvl_{config['mobtep']['mob_checkpoint']}"
    
    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)
    
    
    generate_captions(model, ts_folder_path, metadata_folder_pth, image_folder_path, save_folder_path, batch_size=15, use_chronos=config['mobtep']['use_chronos'])
    
    
    ############################# TOY DEMO ##################################
    """config = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = Mob(chronos_name=config['mobtep']['chronos_name'], internvl_name=config['mobtep']['internvl_name']).to(device)
    checkpoint_path = "/home/ubuntu/thesis/model/checkpoints/Mob2_5-2B_5.174.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    ts = torch.randn(2, 20, 1)
    image_paths = ['/home/ubuntu/thesis/data/samples/plots/air quality_0.jpeg',
                '/home/ubuntu/thesis/data/samples/plots/demography_0.jpeg']

    prompts = ['Describe this line chart about the hourly CO levels in London. Discuss the values you see.', 
                'Describe this line chart about the yearly death rates in Greece. Discuss the values you see.']

    responses = mob_batch_inference(model=model,ts=ts, sum_ts_emb_to=config['mobtep']['sum_ts_emb_to'], image_paths=image_paths, prompts=prompts, max_output_tokens=256)

    print(f"\nResponses:\n")
    for response in responses:
        print("\n", response)"""
    