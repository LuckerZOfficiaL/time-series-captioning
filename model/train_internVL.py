import torch
from transformers import AutoTokenizer, AutoModel
import torchvision.transforms as T
import torch.nn.functional as F
from PIL import Image

from helpers import(
    load_config
)

from internVL import batch_inference

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import os

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




def train_model(model, dataloader, optimizer, ignore_id, epochs=5):
    print("\nStart Training...")
    model.train()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch in range(epochs):
        total_loss = 0
        for pixel_values, input_ids, attention_mask, target_ids, target_attention_mask in tqdm(dataloader):
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

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(dataloader):.4f}")
    return total_loss/len(dataloader) # return the final loss


def compute_loss(logits, labels, ignore_id):
    """
    Compute cross-entropy loss for language modeling.
    """
    shift_logits = logits[:, :-1, :].contiguous()  # Shift so that tokens predict the next token
    shift_labels = labels[:, 1:].contiguous()  # Shift labels accordingly

    loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=ignore_id)
    return loss


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

    dataset = InternVLDataset(tokenizer, 
                                ts_folder=config['path']['ts_folder_path'], 
                                img_folder=config['path']['plot_folder_path'], 
                                metadata_folder=config['path']['metadata_folder_path'], 
                                gt_folder=config['path']['gt_captions_folder_path'],
                                max_tokens=config['mobtep']['max_output_tokens'])
    dataloader = DataLoader(dataset, batch_size=config['train']['batch_size'], shuffle=True)

    """
    img_tensor, input_idx, target_idx = next(iter(dataloader))
    #print(f"Image Tensor: {img_tensor.shape}\nInput Idx: {input_idx.shape}\nTarget Idx: {target_idx.shape}")

    # Decode the first element in input_idx and target_idx back into text
    input_text = tokenizer.decode(input_idx[0])
    target_text = tokenizer.decode(target_idx[0])

    print("Decoded Input Text:", input_text)
    print("Decoded Target Text:", target_text)"""

    ################# TRAINING #################

    optimizer = AdamW(model.parameters(), lr=float(config['train']['lr']), weight_decay=float(config['train']['weight_decay']))

    final_loss = train_model(model, dataloader, optimizer, ignore_id=tokenizer.pad_token_id, epochs=config['train']['epochs'])

    filenpath = f"{config['path']['checkpoints_folder_path']}/internVL2_5-2B_{round(final_loss, 3)}.pth"
    torch.save(model.state_dict(), filenpath)


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
    
