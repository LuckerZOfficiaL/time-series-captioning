import os
import random

from helpers import(
    load_config,
    save_paraphrase_consistency_question
)


def main():
    config = load_config()
    paraphrase_consistency_folder_path = config['path']['paraphrase_consistency_folder_path']
    gt_captions_folder_path = config['path']['gt_captions_folder_path']
    paraphrased_gt_folder_path = config['path']['paraphrased_gt_folder_path']
    num_questions = config['new_tasks']['paraphrase_consistency_questions']
    
    prompt_save_folder = paraphrase_consistency_folder_path + "/" + "prompts"
    answer_save_folder = paraphrase_consistency_folder_path + "/" + "ground truth"
    
    gt_captions_files = os.listdir(gt_captions_folder_path)
    gt_paraphrased_files = os.listdir(paraphrased_gt_folder_path)
    
    
    print(f"{num_questions} questions to be generated, sampling GT captions from {gt_captions_folder_path}.")
    
    for i in range(num_questions):
        random_gt_file = random.choice(gt_captions_files)
        gt_caption_path = os.path.join(gt_captions_folder_path, random_gt_file)
        
        same_phenom = random.choice([True, False])
        
        if same_phenom:
            paraphrase_file = next((f for f in gt_paraphrased_files if f.startswith(random_gt_file[:-4])), None)
            if not paraphrase_file:
                raise FileNotFoundError(f"No paraphrased file found starting with {random_gt_file}")
            paraphrase_path = os.path.join(paraphrased_gt_folder_path, paraphrase_file)
        else:
            paraphrase_file = random.choice(gt_paraphrased_files)
            paraphrase_path = os.path.join(paraphrased_gt_folder_path, paraphrase_file)
        
        print(f"{i+1}/{num_questions} - Generating a {str(same_phenom)} pair for {random_gt_file}.")
        save_paraphrase_consistency_question(caption_path1=gt_caption_path,
                                             caption_path2=paraphrase_path,
                                             same_phenom=same_phenom,
                                             prompt_save_folder=prompt_save_folder,
                                             answer_save_folder=answer_save_folder)
                 
        


if __name__ == "__main__":
    main()