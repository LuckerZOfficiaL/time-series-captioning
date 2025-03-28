from mobtep import CLIP_Mobtep
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
from helpers import(
    load_config,
    pad,
    initialize_bert_model,
    bert_score_loss,
    cross_entropy_loss
)
import torch
from PIL import Image
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



def train(model, train_loader, optimizer, epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    config = load_config()


    for epoch in range(epochs):
        total_loss = 0
        
        for ts_input, text_input, image_input, ground_truth_captions in train_loader:    
            ts_input = ts_input.to(device)

            ground_truth_captions = model.tokenizer(ground_truth_captions, padding=True, truncation=True, return_tensors="pt")["input_ids"].to(device) # strings are converted into indices of tokens
            ground_truth_captions = ground_truth_captions[:, :100]

            print("GT shape: ", ground_truth_captions.shape)
            logits = model(ts_input, text_input, image_input, max_length=config['mobtep']['max_output_tokens'])
            print("Logits shape: ", logits.shape)
            loss = cross_entropy_loss(logits, ground_truth_captions, pad_token_id=model.tokenizer.pad_token_id)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"\nEpoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")


def main():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")

    train_dataset = CaptionDataset(
        ts_paths=["/home/ubuntu/thesis/data/samples/time series/air quality_0.txt",
                "/home/ubuntu/thesis/data/samples/time series/crime_0.txt",
                "/home/ubuntu/thesis/data/samples/time series/demography_0.txt"],
        metadata_paths=["/home/ubuntu/thesis/data/samples/metadata/air quality_0.json",
                        "/home/ubuntu/thesis/data/samples/metadata/crime_0.json",
                        "/home/ubuntu/thesis/data/samples/metadata/demography_0.json"],
        image_paths=["/home/ubuntu/thesis/data/samples/plots/air quality_0.jpeg",
                    "/home/ubuntu/thesis/data/samples/plots/crime_0.jpeg",
                    "/home/ubuntu/thesis/data/samples/plots/demography_0.jpeg"],
        ground_truth_paths=["/home/ubuntu/thesis/data/samples/captions/no external/air quality_0.txt",
                            "/home/ubuntu/thesis/data/samples/captions/no external/crime_0.txt",
                            "/home/ubuntu/thesis/data/samples/captions/no external/demography_0.txt"]
    )

    train_loader = DataLoader(train_dataset, batch_size=config['train']['batch_size'], shuffle=True)

    model = CLIP_Mobtep(tcn_emb_size=config['mobtep']['tcn_emb_size'], 
                        prototype_words=config['mobtep']['anchor_words'], 
                        use_linear_proj=config['mobtep']['use_linear_proj']).to(device)

    optimizer = AdamW(model.parameters(), lr=float(config['train']['lr']))

    train(model, train_loader, optimizer, epochs=config['train']['epochs'])

if __name__ == "__main__":
    main()