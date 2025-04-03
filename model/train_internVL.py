import torch
from transformers import AutoTokenizer, AutoModel
import torchvision.transforms as T
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split, Subset
from PIL import Image

from helpers import(
    load_config
)

from internVL import batch_inference
from tqdm import tqdm
import os
import random

class InternVLDataset(Dataset):
    def __init__(self, tokenizer, ts_folder, img_folder, metadata_folder, gt_folder, input_size=448, max_tokens=250):
        self.tokenizer = tokenizer
        self.ts_path_list = sorted([os.path.join(ts_folder, file) for file in os.listdir(ts_folder)])
        self.img_path_list = sorted([os.path.join(img_folder, file) for file in os.listdir(img_folder)])
        self.metadata_path_list = sorted([os.path.join(metadata_folder, file) for file in os.listdir(metadata_folder)])
        self.gt_path_list = sorted([os.path.join(gt_folder, file) for file in os.listdir(gt_folder)])
        self.input_size = input_size
        self.max_tokens = max_tokens
        self.transform = T.Compose([
            T.Resize((input_size, input_size)),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    def __len__(self):
        return len(self.ts_path_list)

    def __getitem__(self, idx):
        item = {} # this will contain all the information of the idx-th example

        # Read the ts values
        with open(self.ts_path_list[idx], 'r') as file:
            item['ts'] = [float(line.strip()) for line in file]

        # Read the metadata
        metadata_text = ""
        with open(self.metadata_path_list[idx], 'r') as file:
            item['metadata'] = file.read().strip()
        
        item['prompt'] = f"""Provide a description for the following time series, given its line plot and auxiliary metadata: 
            \n
            Time Series:{item['ts']}
            \n
            {item['metadata']}
            \n
            Discuss concrete numbers and the trend. Do not mention any line chart, just directly describe the time series. 
            Give your answer in a single paragraph, without additional explanations or formatting.        
        """       

        # Read grond truth
        with open(self.gt_path_list[idx], 'r') as file:
            item['gt_response'] = file.read().strip()

        image = Image.open(self.img_path_list[idx]).convert("RGB")
        img_tensor = self.transform(image)


        input_text = item["prompt"]
        target_text = item["gt_response"]

        # Tokenize input & output text
        input_encodings = self.tokenizer(
            input_text, return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_tokens
        )
        target_encodings = self.tokenizer(
            target_text, return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_tokens
        )

        # Extract input_ids and attention masks
        input_ids = input_encodings["input_ids"].squeeze(0)
        attention_mask = input_encodings["attention_mask"].squeeze(0)
        target_ids = target_encodings["input_ids"].squeeze(0)
        target_attention_mask = target_encodings["attention_mask"].squeeze(0)

        return img_tensor, input_ids, attention_mask, target_ids, target_attention_mask


def create_dataloaders(tokenizer, ts_folder, img_folder, metadata_folder, gt_folder, train_batch_size=32, val_batch_size=64, val_split=0.2, seed=42, max_tokens=256):
    # Set random seed for reproducibility
    random.seed(seed)
    torch.manual_seed(seed)

    # Create the full dataset
    full_dataset = InternVLDataset(tokenizer, ts_folder, img_folder, metadata_folder, gt_folder, max_tokens=max_tokens)
    
    # Shuffle all indices randomly
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    random.shuffle(indices)  # Randomly shuffle indices

    # Split indices into train and val
    val_size = int(dataset_size * val_split)
    train_indices, val_indices = indices[val_size:], indices[:val_size]  # First part = val, rest = train

    # Create train and val datasets using Subset
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    #print(f"train dataset {len(train_dataset)}, val dataset {len(val_dataset)}")

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader


def compute_loss(logits, labels, ignore_id):
    """
    Compute cross-entropy loss for language modeling.
    """
    shift_logits = logits[:, :-1, :].contiguous()  # Shift so that tokens predict the next token
    shift_labels = labels[:, 1:].contiguous()  # Shift labels accordingly

    loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=ignore_id)
    return loss

    

def train_model(model, train_loader, optimizer, ignore_id, epochs=5, val_loader=None): # if val_loader is not None, it also does validation
    print("\nStart Training...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        total_train_loss = 0
        model.train()
        for pixel_values, input_ids, attention_mask, target_ids, target_attention_mask in tqdm(train_loader):
            pixel_values = pixel_values.cuda().to(torch.bfloat16)
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
            target_ids = target_ids.cuda()
            #target_attention_mask = target_attention_mask.cuda() # useless

            outputs = model(pixel_values=pixel_values, 
                            input_ids=input_ids,
                            labels=target_ids,
                            image_flags=torch.ones(input_ids.size(0)),
                            attention_mask=attention_mask)
            loss = compute_loss(outputs.logits, target_ids, ignore_id=ignore_id)
            #loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {total_train_loss/len(train_loader):.4f}")
        train_losses.append(total_train_loss/len(train_loader))
        
        # Validation
        if val_loader is not None:
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for val_pixel_values, val_input_ids, val_attention_mask, val_target_ids, val_target_attention_mask in tqdm(val_loader):
                    val_pixel_values = val_pixel_values.cuda().to(torch.bfloat16)
                    val_input_ids = val_input_ids.cuda()
                    val_attention_mask = val_attention_mask.cuda()
                    val_target_ids = val_target_ids.cuda()
                    val_outputs = model(pixel_values=val_pixel_values, 
                                        input_ids=val_input_ids,
                                        labels=val_target_ids,
                                        image_flags=torch.ones(val_input_ids.size(0)),
                                        attention_mask=val_attention_mask)
                    val_loss = compute_loss(val_outputs.logits, val_target_ids, ignore_id=ignore_id)
                    total_val_loss += val_loss.item()
            print(f"Epoch {epoch+1}/{epochs} - Validation Loss: {total_val_loss/len(val_loader):.4f}")
            val_losses.append(total_val_loss/len(val_loader))

    return train_losses, val_losses




def main():
    config = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path = config['mobtep']['internvl_name']
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        #low_cpu_mem_usage=True,
        #use_flash_attn=True,
        trust_remote_code=True).to(device)

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

    train_loader, val_loader = create_dataloaders(tokenizer, 
                                                    ts_folder=config['path']['ts_folder_path'], 
                                                    img_folder=config['path']['plot_folder_path'], 
                                                    metadata_folder=config['path']['metadata_folder_path'], 
                                                    gt_folder=config['path']['gt_captions_folder_path'],
                                                    max_tokens=config['mobtep']['max_output_tokens'],
                                                    train_batch_size=config['train']['batch_size'],
                                                    val_batch_size=config['eval']['batch_size'],
                                                    val_split=config['eval']['val_split'],
                                                    seed=config['general']['random_seed'])

    ####################################### TRAINING #######################################
    optimizer = AdamW(model.parameters(), lr=float(config['train']['lr']), weight_decay=float(config['train']['weight_decay']))

    train_losses, val_losses = train_model(model, 
                            train_loader=train_loader, 
                            optimizer=optimizer, 
                            ignore_id=tokenizer.pad_token_id, 
                            epochs=config['train']['epochs'],
                            val_loader=val_loader)
    print("\nTrain Losses: ", train_losses)
    print("\nVal Losses: ", val_losses)


    ######################################## SAVING CHECKPOINT #######################################
    filepath = f"{config['path']['checkpoints_folder_path']}/internVL2_5-2B_{round(val_losses[-1], 3) if val_losses != [] else ""}.pth"
    torch.save(model.state_dict(), filepath)


    ####################################### TOY DEMO #######################################
    ts = torch.randn(2, 20, 1)
    image_paths = ['/home/ubuntu/thesis/data/samples/plots/air quality_0.jpeg',
                '/home/ubuntu/thesis/data/samples/plots/demography_0.jpeg']

    prompts = ['Describe this line chart about the hourly CO levels in London. Discuss the values you see.', 
                'Describe this line chart about the yearly death rates in Greece. Discuss the values you see.']

    responses = batch_inference(model, tokenizer, image_paths, prompts, max_output_tokens=256)

    print(f"\nResponses:\n")
    for response in responses:
        print("\n", response)



if __name__ == "__main__":
    main()
    
