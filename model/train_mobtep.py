from mobtep import CLIP_Mobtep
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
from helpers import(
    load_config,
    pad,
    cross_entropy_loss
)
import torch
from PIL import Image
import os
import os

class CaptionDataset(torch.utils.data.Dataset):
    def __init__(self, ts_paths, metadata_paths, image_paths, ground_truth_paths, image_transform=None):
        """
        Args:
            time_series_paths (list of str): List of file paths to time series data files (e.g., CSV files).
            text_paths (list of str): List of file paths to text files containing metadata or text data.
            image_paths (list of str): List of file paths to image files.
            ground_truth_captions (list of str): List of ground truth captions for each sample.
            time_series_transform (callable, optional): A function/transform to apply to the time series data.
            image_transform (callable, optional): A function/transform to apply to the image data.
        """
        self.ts_paths = ts_paths
        self.metadata_paths = metadata_paths
        self.image_paths = image_paths
        self.ground_truth_paths = ground_truth_paths

        self.max_ts_len = 0
        for ts_path in self.ts_paths: #compute the maximum ts length in the dataset, this value is used for padding
            with open(ts_path, 'r') as file:
                ts_input = [float(line.strip()) for line in file]
            self.max_ts_len = max(self.max_ts_len, len(ts_input))

        if image_transform is None:
            self.image_transform = transforms.ToTensor()
        else:
            self.image_transform = image_transform

    def __len__(self):
        return len(self.ts_paths)

    def __getitem__(self, idx):
        with open(self.ts_paths[idx], 'r') as file:
            ts_input = [float(line.strip()) for line in file]
            ts_input = torch.tensor(ts_input)
            ts_input = pad(ts_input, max_len=self.max_ts_len, with_value=0)
            ts_input = ts_input.unsqueeze(-1)
            

        with open(self.metadata_paths[idx], 'r') as file:
            metadata = file.read()
        
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = self.image_transform(image)

        with open(self.ground_truth_paths[idx], 'r') as file:
            ground_truth_caption = file.read()      
        
        text_input = f"""
            Here is a time series line chart and its metadata:
            \n
            {metadata}
        """

        return ts_input, text_input, image, ground_truth_caption



def train(model, train_loader, val_loader=None, optimizer=None, epochs=5, milestones=None, early_stopping=False, 
          patience=None, clip_grad_norm=None, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
    model.to(device)
    model.train()
    config = load_config()

    if milestones is not None:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        total_loss = 0
        model.train()
        
        # Training loop
        for ts_input, text_input, image_input, ground_truth_captions in train_loader:    
            ts_input = ts_input.to(device)
            image_input = image_input.to(device)
            
            ground_truth_captions_ids = model.tokenizer(
                ground_truth_captions, 
                padding=True, 
                truncation=True, 
                return_tensors="pt"
            )["input_ids"].to(device)
            
            if config['train']['teacher_forcing']:
                loss = model(
                    ts_input, 
                    text_input, 
                    image_input,  
                    teacher_forcing=True,
                    ground_truth_texts=ground_truth_captions
                )
            else:
                logits = model(
                    ts_input, 
                    text_input, 
                    image_input, 
                    teacher_forcing=False, 
                    max_length=config['mobtep']['max_output_tokens']
                )
                loss = cross_entropy_loss(
                    logits, 
                    ground_truth_captions_ids, 
                    pad_token_id=model.tokenizer.pad_token_id
                )

            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}")
        
        # Validation loop
        if val_loader is not None:
            model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for ts_input, text_input, image_input, ground_truth_captions in val_loader:
                    ts_input = ts_input.to(device)
                    image_input = image_input.to(device)
                    
                    ground_truth_captions_ids = model.tokenizer(
                        ground_truth_captions, 
                        padding=True, 
                        truncation=True, 
                        return_tensors="pt"
                    )["input_ids"].to(device)
                    
                    # Use the same loss calculation as in training
                    if config['train']['teacher_forcing']:
                        loss = model(
                            ts_input, 
                            text_input, 
                            image_input,  
                            teacher_forcing=True,
                            ground_truth_texts=ground_truth_captions
                        )
                    else:
                        logits = model(
                            ts_input, 
                            text_input, 
                            image_input, 
                            teacher_forcing=False, 
                            max_length=config['mobtep']['max_output_tokens']
                        )
                        loss = cross_entropy_loss(
                            logits, 
                            ground_truth_captions_ids, 
                            pad_token_id=model.tokenizer.pad_token_id
                        )
                    
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            print(f"Epoch {epoch + 1}/{epochs}, Val Loss: {avg_val_loss:.4f}")
            
            # Early stopping
            if early_stopping:
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    # Save the best model
                    torch.save(model.state_dict(), 'best_model.pt')
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch + 1}")
                        break
        
        if milestones is not None:
            scheduler.step()


def main():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")

    # Reading Data
    ts_folder = "/home/ubuntu/thesis/data/samples/time series/"
    ts_paths = [os.path.join(ts_folder, file) for file in os.listdir(ts_folder)]

    metadata_folder = "/home/ubuntu/thesis/data/samples/metadata/"
    metadata_paths = [os.path.join(metadata_folder, file) for file in os.listdir(metadata_folder)]

    image_folder = "/home/ubuntu/thesis/data/samples/plots/"
    image_paths = [os.path.join(image_folder, file) for file in os.listdir(image_folder)]

    grond_truth_folder = "/home/ubuntu/thesis/data/samples/captions/no external"
    ground_truth_paths = [os.path.join(grond_truth_folder, file) for file in os.listdir(grond_truth_folder)]

    ts_paths.sort()
    metadata_paths.sort()
    image_paths.sort()
    ground_truth_paths.sort()


    train_dataset = CaptionDataset(
        ts_paths=ts_paths,
        metadata_paths=metadata_paths,
        image_paths=image_paths,
        ground_truth_paths=ground_truth_paths
    )

    train_loader = DataLoader(train_dataset, 
                                batch_size=config['train']['batch_size'], shuffle=True)

    model = CLIP_Mobtep(tcn_emb_size=config['mobtep']['tcn_emb_size'], 
                        prototype_words=config['mobtep']['anchor_words'], 
                        use_linear_proj=config['mobtep']['use_linear_proj']).to(device)

    optimizer = AdamW(model.parameters(), lr=float(config['train']['lr']))

    train(model, 
        train_loader=train_loader, 
        val_loader=None,
        optimizer=optimizer, 
        epochs=config['train']['epochs'], 
        milestones=config['train']['milestones'],
        early_stopping=config['train']['early_stopping'],
        patience=config['train']['patience'],
        clip_grad_norm=config['train']['clip_grad_norm']
        )

    ts_input = torch.randn(3, 100, 1).to(device)  # Example time series (3 samples, length 100)
    text_input = ["Here's a time series describing hourly air quality.",
                "Here's a time series describing daily crimes.",
                "Here's a time series describing yearly."
    ]
    
    img_paths = ['/home/ubuntu/thesis/data/samples/plots/air quality_0.jpeg', 
                    '/home/ubuntu/thesis/data/samples/plots/crime_0.jpeg', 
                    '/home/ubuntu/thesis/data/samples/plots/demography_0.jpeg']

    images = [Image.open(img_path).convert("RGB") for img_path in img_paths]
    transform = transforms.ToTensor()
    images = [transform(image) for image in images]

    prompt_input = ["Provide a time series description."]*3

    mobtep = CLIP_Mobtep(tcn_emb_size=128, prototype_words=config['mobtep']['anchor_words'], use_linear_proj=False).to(device)
    mobtep.eval()

    
    with torch.no_grad():
        output = mobtep.generate_captions(ts_input, text_input, images, prompt_input,
                                            max_length=config['mobtep']['max_output_tokens'],)
    for caption in output:
        print("\n\n", caption)
    

if __name__ == "__main__":
    main()