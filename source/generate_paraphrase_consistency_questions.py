import os
import random

from helpers import(
    load_config,
    save_paraphrase_consistency_question,
    perturb_caption
)


def main():
    config = load_config()
    paraphrase_consistency_folder_path = config['path']['paraphrase_consistency_folder_path']
    gt_captions_folder_path = config['path']['gt_captions_folder_path']
    paraphrased_gt_folder_path = config['path']['paraphrased_gt_folder_path']
    #num_questions = config['new_tasks']['paraphrase_consistency_questions']
    
    prompt_save_folder = paraphrase_consistency_folder_path + "/" + "prompts"
    answer_save_folder = paraphrase_consistency_folder_path + "/" + "ground truth"
    
    gt_captions_files = os.listdir(gt_captions_folder_path)
    gt_paraphrased_files = os.listdir(paraphrased_gt_folder_path)
    

    num_existing_files = len(os.listdir(prompt_save_folder))
    #num_questions = max(0, num_questions-num_existing_files)
    num_questions = len(gt_captions_files)
    
    
    print(f"Number of questions already created: {num_existing_files}")
    print(f"{num_questions-num_existing_files} questions yet to be generated, sampling GT captions from {gt_captions_folder_path}.")
    
    #for i in range(num_questions):
    for i, gt_filename in enumerate(gt_captions_files):
        #gt_filename = random.choice(gt_captions_files)
        #while gt_filename in prompt_save_folder:
        #    gt_filename = random.choice(gt_captions_files)# if the same gt caption was already used in a previous run of this script, random sample again
        

        gt_caption_path = os.path.join(gt_captions_folder_path, gt_filename)
        
        same_phenom = random.choice([True, False])
        
        if same_phenom:
            paraphrase_file = next((f for f in gt_paraphrased_files if f.startswith(gt_filename[:-4])), None)
            if not paraphrase_file:
                raise FileNotFoundError(f"No paraphrased file found starting with {gt_filename}")
            paraphrase_path = os.path.join(paraphrased_gt_folder_path, paraphrase_file)
            with open(paraphrase_path, 'r') as file:
                caption2 = file.read()
        else:
            with open(gt_caption_path, 'r') as file:
                gt_caption = file.read()
            caption2 = perturb_caption(gt_caption, model=config['model']['refinement_model']) # create the second caption as the perturbed version of the gt caption, this makes the task harder
        
        print(f"{i+1}/{num_questions-num_existing_files} - Generating a {str(same_phenom)} pair for {gt_filename}.")
        save_paraphrase_consistency_question(caption_path1=gt_caption_path,
                                             caption2=caption2,
                                             same_phenom=same_phenom,
                                             prompt_save_folder=prompt_save_folder,
                                             answer_save_folder=answer_save_folder)
                 
        


if __name__ == "__main__":
    main()