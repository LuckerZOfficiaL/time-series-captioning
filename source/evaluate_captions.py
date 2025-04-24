import os
from transformers import BertForSequenceClassification, BertTokenizer
from bert_score import score
import json

from helpers import(
    load_config,
    numeric_score,
    oracle_score,
    bleu_score,
    rouge_score,
    meteor_score,
    save_file,
    extract_num_dict_from_dict,
    extract_num_dict_from_text,
    compare_num_dicts
    
)


def get_batch_dict_num_score(generated_captions, gt_metadata):
    result_dict = {
        "minimum": 0,
        "maximum": 0,
        "mean": 0,
        "std": 0,
        "minimum_nulls": 0,
        "maximum_nulls": 0,
        "mean_nulls": 0,
        "std_nulls": 0,
    }

    failed_extractions = 0

    for gen_capt, gt_metadata in zip(generated_captions, gt_metadata):
        
        try:
            num_dict_gen = extract_num_dict_from_text(gen_capt)
            #print("\nGen dict: ", num_dict_gen)
            num_dict_gt = extract_num_dict_from_dict(gt_metadata)
            #print("\nGT dict: ", num_dict_gt)
            comparison = compare_num_dicts(num_dict_gen, num_dict_gt)
            #print(comparison)
            if comparison['minimum'] == None:
                result_dict['minimum_nulls'] += 1
            else:
                result_dict['minimum'] += comparison['minimum']
                
            if comparison['maximum'] == None:
                result_dict['maximum_nulls'] += 1
            else:
                result_dict['maximum'] += comparison['maximum']

            if comparison['mean'] == None:
                result_dict['mean_nulls'] += 1
            else:
                result_dict['mean'] += comparison['mean']

            if comparison['std'] == None:
                result_dict['std_nulls'] += 1
            else:
                result_dict['std'] += comparison['std']
        except Exception as e:
            failed_extractions += 1   
            print(f"Extraction failed, this is occurrence {failed_extractions}.")
            
    #print(result_dict)
       
    result_dict['minimum'] /= (len(generated_captions) - result_dict["minimum_nulls"])
    result_dict['maximum'] /= (len(generated_captions) - result_dict["maximum_nulls"])
    
    if len(generated_captions) - result_dict["mean_nulls"] != 0:
        result_dict['mean'] /= (len(generated_captions) - result_dict["mean_nulls"]) 
    else:
        result_dict['mean'] = None
    if len(generated_captions) - result_dict["std_nulls"] != 0:
        result_dict['std'] /= (len(generated_captions) - result_dict["std_nulls"])
    else:
        result_dict['std'] = None
    
    #del result_dict['minimum_nulls']
    #del result_dict['maximum_nulls']
    #del result_dict['mean_nulls']
    #del result_dict['std_nulls']
       
    return result_dict
    
    

def get_batch_score(generated_captions, gt_captions, score_function):
    score_sum = 0
    for gen_capt, gt_capt in zip(generated_captions, gt_captions):
        score_sum += score_function(gen_capt, gt_capt)
    return score_sum / len(generated_captions)


def main():
    config = load_config()
    eval_model = config['path']['generated_captions_folder_path'].split("/")[-1].replace(" ", "_")
    #eval_model = "gemini-2.0-flash_text" # _text
    
    print("\nEvaluating captions from: ", eval_model)

    generated_captions_folder_path = config['path']['generated_captions_folder_path']
    #generated_captions_folder_path = f"/home/ubuntu/thesis/data/samples/len 10/generated captions/{eval_model}"
    generated_caption_paths = [os.path.join(generated_captions_folder_path, filename) for filename in os.listdir(generated_captions_folder_path)]
    generated_caption_paths.sort()

    gt_captions_folder_path = config['path']['gt_captions_folder_path']
    #gt_captions_folder_path = "/home/ubuntu/thesis/data/samples/len 10/captions"
    gt_caption_paths = [os.path.join(gt_captions_folder_path, filename) for filename in os.listdir(gt_captions_folder_path)]
    gt_caption_paths.sort()
    
    gt_metadata_folder_path = config['path']['gt_metadata_folder_path']
    gt_metadata_paths = [os.path.join(gt_metadata_folder_path, filename) for filename in os.listdir(gt_metadata_folder_path)]
    gt_metadata_paths.sort()

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
    
    dataset_gt_metadata_paths = {}
    for gt_path in gt_metadata_paths:
        dataset_name = gt_path.split("/")[-1].split('_')[0]
        if dataset_name not in dataset_gt_metadata_paths:
            dataset_gt_metadata_paths[dataset_name] = []
        dataset_gt_metadata_paths[dataset_name].append(gt_path)
        
        
    
    
    for dataset_name in dataset_gt_caption_paths:
        dataset_gt_caption_paths[dataset_name].sort()
    
    for dataset_name in dataset_generated_caption_paths:
        dataset_generated_caption_paths[dataset_name].sort()
    
    for dataset_name in dataset_gt_metadata_paths:
        dataset_gt_metadata_paths[dataset_name].sort()
          

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

    gt_metadatas = {}
    for dataset in dataset_gt_metadata_paths.keys():
        for gt_metadata_path in dataset_gt_metadata_paths[dataset]:
            with open(gt_metadata_path, 'r') as file:
                gt_metadata = json.load(file)
                if dataset not in gt_metadatas:
                    gt_metadatas[dataset] = []
                gt_metadatas[dataset].append(gt_metadata)

    
    save_path = config['path']['evaluation_results_folder_path'] + "/" + eval_model + ".json"
    #save_path = f"/home/ubuntu/thesis/data/samples/len 10/evaluation results/{eval_model}.json"

    if os.path.exists(save_path):
        print(f"Evaluation results already exist at {save_path}. Loading existing results...")
        with open(save_path, 'r') as file:
            result_dict = json.load(file)
    else:   
        print(f"Creating result dictionary from scratch...")
        result_dict = {}
        for dataset in dataset_generated_caption_paths:
                result_dict[dataset] = {}
            
    for dataset in dataset_gt_caption_paths.keys():
    #for dataset in ["demography"]:
        if result_dict[dataset] != {}:
            continue
        
        gen_capts = generated_captions[dataset]
        gt_capts = gt_captions[dataset]
        gt_metadata = gt_metadatas[dataset]
    
        print(f"\n\n{dataset}: {len(gen_capts)} captions are being scored...")
        
        
        ################################# Numeric Score 2.0 ###############################################
        
        num_score_dict = get_batch_dict_num_score(generated_captions=gen_capts, gt_metadata=gt_metadata)
        
        print(f"NUMERIC SCORE 2.0: {num_score_dict}")
        
        result_dict[dataset]['numeric score 2.0'] = num_score_dict
        
        ################################# BERT SCORE ##############################################
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
        ################################# Numeric Score ###############################################
        
        num_score = get_batch_score(generated_captions=gen_capts, gt_captions=gt_capts, score_function=numeric_score)
        
        print(f"NUMERIC SCORE: {round(num_score, 3)}")
        
        result_dict[dataset]['numeric score'] = round(num_score, 3)
        
        
        ################################# BLEU SCORE ###############################################
        
        bleu = get_batch_score(generated_captions=gen_capts, gt_captions=gt_capts, score_function=bleu_score)
        
        print(f"BLEU SCORE: {round(bleu, 3)}")
        
        result_dict[dataset]['bleu score'] = round(bleu, 3)
        
        ################################# ROUGE SCORE ###############################################
        
        rouge = get_batch_score(generated_captions=gen_capts, gt_captions=gt_capts, score_function=rouge_score)
        
        print(f"ROUGE SCORE: {round(rouge, 3)}")
        
        result_dict[dataset]['rouge score'] = round(rouge, 3)
        
        ################################# METEOR SCORE ###############################################
        
        meteor = get_batch_score(generated_captions=gen_capts, gt_captions=gt_capts, score_function=meteor_score)
        
        print(f"METEOR SCORE: {round(rouge, 3)}")
        
        result_dict[dataset]['meteor score'] = round(meteor, 3)
        
        
        ################################# ORACLE SCORE ###############################################
        
        oracle_sc = get_batch_score(generated_captions=gen_capts, gt_captions=gt_capts, score_function=oracle_score)
        oracle_sc = round(oracle_sc/100, 3)
        print(f"ORACLE SCORE: {round(oracle_sc, 3)}")
        
        result_dict[dataset]['oracle score'] = round(oracle_sc, 3)
        
        
        ################################ SAVE CHECKPOINT ###############################################
        save_file(result_dict, filepath=save_path)
        
        
    ############################### AVERAGE SCORE #############################################
    
    average_scores = {
        "bert score": {
            "precision": 0,
            "recall": 0,
            "f1": 0
            },
        "numeric score 2.0":{
            "minimum": 0,
            "maximum": 0,
            "mean": 0,
            "std": 0
        },
        "numeric score": 0,
        "bleu score": 0,
        "rouge score": 0,
        "meteor score": 0,
        "oracle score": 0
    }

    # Don't include 'average' in the dataset count or loop
    datasets = [d for d in result_dict if d != "average"]
    dataset_count = len(datasets)
    mean_nones = 0
    std_nones = 0

    for dataset in datasets:
        average_scores["bert score"]["precision"] += result_dict[dataset]["bert score"]["precision"]
        average_scores["bert score"]["recall"] += result_dict[dataset]["bert score"]["recall"]
        average_scores["bert score"]["f1"] += result_dict[dataset]["bert score"]["f1"]
        
        average_scores["numeric score 2.0"]["minimum"] += result_dict[dataset]["numeric score 2.0"]["minimum"]
        average_scores["numeric score 2.0"]["maximum"] += result_dict[dataset]["numeric score 2.0"]["maximum"]
        
        if result_dict[dataset]["numeric score 2.0"]["mean"] != None:
            average_scores["numeric score 2.0"]["mean"] += result_dict[dataset]["numeric score 2.0"]["mean"]
        else:
            mean_nones += 1
            
        if result_dict[dataset]["numeric score 2.0"]["std"] != None:
            average_scores["numeric score 2.0"]["std"] += result_dict[dataset]["numeric score 2.0"]["std"]
        else:
            std_nones += 1
        
        average_scores["numeric score"] += result_dict[dataset]["numeric score"]
        average_scores["bleu score"] += result_dict[dataset]["bleu score"]
        average_scores["rouge score"] += result_dict[dataset]["rouge score"]
        average_scores["meteor score"] += result_dict[dataset]["meteor score"]
        average_scores["oracle score"] += result_dict[dataset]["oracle score"]

    # Compute the mean for each score
    average_scores["bert score"]["precision"] /= dataset_count
    average_scores["bert score"]["recall"] /= dataset_count
    average_scores["bert score"]["f1"] /= dataset_count
    average_scores["numeric score 2.0"]["minimum"] /= dataset_count
    average_scores["numeric score 2.0"]["maximum"] /= dataset_count
    average_scores["numeric score 2.0"]["mean"] /= dataset_count-mean_nones
    average_scores["numeric score 2.0"]["std"] /= dataset_count-std_nones
    average_scores["numeric score"] /= dataset_count
    average_scores["bleu score"] /= dataset_count
    average_scores["rouge score"] /= dataset_count
    average_scores["meteor score"] /= dataset_count
    average_scores["oracle score"] /= dataset_count

    # Now safe to modify the dictionary
    result_dict["average"] = average_scores

    # Save the result
    save_file(result_dict, filepath=save_path)
            
        
    
if __name__ == "__main__":
    main()