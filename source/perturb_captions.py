import os

from helpers import(
    perturb_semantically,
    perturb_numerically,
    load_config
)


def main():
    config = load_config()
    gt_captions_folder_path = config['path']['gt_captions_folder_path']
    numerically_perturbed_folder_path = "/home/ubuntu/thesis/data/samples/new samples no overlap/test/gt_captions_numerically_perturbed"
    semantically_perturbed_folder_path = "/home/ubuntu/thesis/data/samples/new samples no overlap/test/gt_captions_semantically_perturbed"
    
    os.makedirs(numerically_perturbed_folder_path, exist_ok=True)
    os.makedirs(semantically_perturbed_folder_path, exist_ok=True)
    
    gt_caption_filenames = os.listdir(gt_captions_folder_path)
    
    done_captions = len(os.listdir(numerically_perturbed_folder_path))
    
    for i, filename in enumerate(gt_caption_filenames):
        if filename in os.listdir(numerically_perturbed_folder_path):
            continue
        file_path = os.path.join(gt_captions_folder_path, filename)
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                gt_caption = file.read()
                
            num_perturbed = perturb_numerically(caption=gt_caption, model=config['model']['refinement_model'])
            sem_perturbed = perturb_semantically(caption=gt_caption, model=config['model']['refinement_model'])
            
            with open(os.path.join(numerically_perturbed_folder_path, filename), 'w', encoding='utf-8') as num_file:
                num_file.write(num_perturbed)

            with open(os.path.join(semantically_perturbed_folder_path, filename), 'w', encoding='utf-8') as num_file:
                num_file.write(sem_perturbed)
                
            if i % 20 == 19:
                print(f"{done_captions+i+1}/{len(gt_caption_filenames)} Done.")
            

if __name__ == "__main__":
    main()