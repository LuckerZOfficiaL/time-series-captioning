import os
import random
import shutil
from pathlib import Path




def mix_captions():
    # Define source folders
    source_folders = [
        "/home/ubuntu/thesis/data/samples/new_samples_no_overlap/test/gt_captions",
        "/home/ubuntu/thesis/data/samples/new_samples_no_overlap/test/second_domains_paraphrased/gemma",
        "/home/ubuntu/thesis/data/samples/new_samples_no_overlap/test/second_domains_paraphrased/llama",
        "/home/ubuntu/thesis/data/samples/new_samples_no_overlap/test/second_domains_paraphrased/gpt-4o"
    ]
    
    # Define destination folder
    dest_folder = "/home/ubuntu/thesis/data/samples/new_samples_no_overlap/test/human_edited_captions/second_domains/unedited_mixed_captions"
    
    # Create destination folder if it doesn't exist
    os.makedirs(dest_folder, exist_ok=True)
    
    # Get all filenames from the last folder (assuming last folder contains the target samples)
    first_folder = Path(source_folders[-1])
    filenames = [f.name for f in first_folder.iterdir() if f.is_file()]
    
    
    # For each filename, randomly select a source folder and copy the file
    for filename in filenames:
        # Randomly select one of the four source folders (25% probability each)
        probabilities = [0.29, 0.13, 0.29, 0.29]
        selected_folder = random.choices(source_folders, weights=probabilities, k=1)[0]
        #selected_folder = random.choice(source_folders)
        
        # Construct source and destination paths
        source_path = os.path.join(selected_folder, filename)
        dest_path = os.path.join(dest_folder, filename)
        
        # Copy the file
        if os.path.exists(source_path):
            shutil.copy2(source_path, dest_path)
            print(f"Copied {filename} from {os.path.basename(selected_folder)}")
        else:
            print(f"Warning: {filename} not found in {selected_folder}")

if __name__ == "__main__":
    mix_captions()