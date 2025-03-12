import torch
import torchvision.transforms as transforms
from PIL import Image
from timm import create_model
import os
from glob import glob

class ViTEncoder(torch.nn.Module):
    def __init__(self, model_name="vit_base_patch16_224", pretrained=True):
        super(ViTEncoder, self).__init__()
        # Load the pretrained ViT model
        self.model = create_model(model_name, pretrained=pretrained)
        self.model.head = torch.nn.Identity()  # Remove classification head

    def forward(self, x):
        return self.model(x)



def main():
    # Define preprocessing pipeline for line chart images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


    def preprocess_images(path):
        """
        If the given path is a file, process and return a single image tensor.
        If the given path is a folder, process all images in the folder and return a batch tensor.
        """
        if os.path.isfile(path):  # Single image case
            image = Image.open(path).convert("RGB")
            return transform(image).unsqueeze(0)  # Shape: [1, C, H, W]
        
        elif os.path.isdir(path):  # Folder case
            image_paths = glob(os.path.join(path, "*.png")) + glob(os.path.join(path, "*.jpg")) + glob(os.path.join(path, "*.jpeg"))
            images = [transform(Image.open(img).convert("RGB")) for img in image_paths]
            
            if not images:
                raise ValueError("No valid images found in the folder.")
            
            return torch.stack(images)  # Shape: [B, C, H, W]
        
        else:
            raise ValueError("Invalid path: not a file or folder.")

    # Load encoder
    vit_encoder = ViTEncoder()
    vit_encoder.eval()

    # Example usage
    image_tensor = preprocess_images("/home/ubuntu/thesis/data/samples/plots/demography_0.jpeg") # [batch size, channels, height, width]
    print("\nInput shape: ", image_tensor.shape)
    with torch.no_grad():
        embedding = vit_encoder(image_tensor)
    print(embedding.shape)  # Should be [1, 768] for ViT-B/16


if __name__ == "__main__":
    main()