import os
from transformers import BertForSequenceClassification, BertTokenizer
from bert_score import score

from helpers import(
    load_config,
    numeric_score,
    oracle_score,
    bleu_score,
    rouge_score,
    save_file
    
)

def get_batch_score(generated_captions, gt_captions, score_function):
    score_sum = 0
    for gen_capt, gt_capt in zip(generated_captions, gt_captions):
        score_sum += score_function(gen_capt, gt_capt)
    return score_sum / len(generated_captions)


def main():
    config = load_config()
    eval_model = config['path']['generated_captions_folder_path'].split("/")[-1].replace(" ", "_")

    print("\nEvaluating captions from: ", eval_model)

    generated_captions_folder_path = config['path']['generated_captions_folder_path']
    generated_caption_paths = [os.path.join(generated_captions_folder_path, filename) for filename in os.listdir(generated_captions_folder_path)]
    generated_caption_paths.sort()

    gt_captions_folder_path = config['path']['gt_captions_folder_path']
    gt_caption_paths = [os.path.join(gt_captions_folder_path, filename) for filename in os.listdir(gt_captions_folder_path)]
    gt_caption_paths.sort()

    assert len(generated_caption_paths) == len(gt_caption_paths)
    for gen_path, gt_path in zip(generated_caption_paths, gt_caption_paths): # checking that the caption paths between generated and gt are aligned
        if gen_path.split("/")[-1] != gt_path.split("/")[-1]:
            print("\n\nCaption filenames are not aligned between the two folders!")
            exit()
    
    # truncate the samples for quick code testing, remove these 2 lines in official evaluation
    #generated_caption_paths = generated_caption_paths[:100]
    #gt_caption_paths = gt_caption_paths[:100]

    # Create a dictionary to group paths by dataset name
    dataset_gt_caption_paths = {}
    for gt_path in gt_caption_paths:
        dataset_name = gt_path.split("/")[-1].split('_')[0]
        if dataset_name not in dataset_gt_caption_paths:
            dataset_gt_caption_paths[dataset_name] = []
        dataset_gt_caption_paths[dataset_name].append(gt_path)
        
    dataset_generated_caption_paths = {}
    for generated_path in generated_caption_paths:
        dataset_name = generated_path.split("/")[-1].split('_')[0]
        if dataset_name not in dataset_generated_caption_paths:
            dataset_generated_caption_paths[dataset_name] = []
        dataset_generated_caption_paths[dataset_name].append(generated_path)
    
    
    for dataset_name in dataset_gt_caption_paths:
        dataset_gt_caption_paths[dataset_name].sort()
    
    for dataset_name in dataset_generated_caption_paths:
        dataset_generated_caption_paths[dataset_name].sort()
          

    # Read the captions into lists of strings
    generated_captions = {}
    for dataset in dataset_generated_caption_paths.keys():
        for generated_caption_path in dataset_generated_caption_paths[dataset]:
            with open(generated_caption_path, 'r') as file:
                generated_caption = file.read()
                if dataset not in generated_captions:
                    generated_captions[dataset] = []
                generated_captions[dataset].append(generated_caption)

    gt_captions = {}
    for dataset in dataset_gt_caption_paths.keys():
        for gt_caption_path in dataset_gt_caption_paths[dataset]:
            with open(gt_caption_path, 'r') as file:
                gt_caption = file.read()
                if dataset not in gt_captions:
                    gt_captions[dataset] = []
                gt_captions[dataset].append(gt_caption)


    for dataset in dataset_gt_caption_paths.keys():
        gen_capts = generated_captions[dataset]
        gt_capts = gt_captions[dataset]
    
        print(f"\n\n{dataset}: {len(gen_capts)} captions are being scored...")
        
        
        result_dict = {}
        for dataset in dataset_generated_caption_paths:
            result_dict[dataset] = {}
        
        ############################### BERT SCORE #############################################
        
        """
            P (Precision): Measures how much of the candidate text's meaning is captured in the reference text.
            R (Recall): Measures how much of the reference text's meaning is captured in the candidate text.
            F1 (F1-score): The harmonic mean of precision and recall, providing a balanced similarity measure.
        """
        P, R, F1 = score(gen_capts, gt_capts, lang="en", model_type=config['eval']['bertscore_model'])

        p_mean = sum(P) / len(P)
        r_mean = sum(R) / len(R)
        f1_mean = sum(F1) / len(F1)
        
        print(f"BERT SCORE: Mean P: {round(p_mean.item(), 3)}, Mean R: {round(r_mean.item(), 3)}, Mean F1: {round(f1_mean.item(), 3)}")

        result_dict[dataset]['bert score'] = {
            "precision": round(p_mean.item(),3),
            "recall": round(r_mean.item(), 3),
            "f1": round(f1_mean.item(), 3)
        }
        ############################### Numeric Score #############################################
        
        num_score = get_batch_score(generated_captions=gen_capts, gt_captions=gt_capts, score_function=numeric_score)
        
        print(f"NUMERIC SCORE: {round(num_score, 3)}")
        
        result_dict[dataset]['numeric score'] = round(num_score, 3)
        
        ############################### BLEU SCORE #############################################
        
        bleu = get_batch_score(generated_captions=gen_capts, gt_captions=gt_capts, score_function=bleu_score)
        
        print(f"BLEU SCORE: {round(bleu, 3)}")
        
        result_dict[dataset]['bleu score'] = round(bleu, 3)
        
        ############################### ROUGE SCORE #############################################
        
        rouge = get_batch_score(generated_captions=gen_capts, gt_captions=gt_capts, score_function=rouge_score)
        
        print(f"ROUGE SCORE: {round(rouge, 3)}")
        
        result_dict[dataset]['rouge score'] = round(rouge, 3)
        
        
        
    print("\n\n", result_dict)
    save_path = config['path']['evaluation_results_folder_path']+"/"+eval_model+".json"
    print(save_path)
    save_file(result_dict, filepath=save_path)
        
        
    
if __name__ == "__main__":
    main()