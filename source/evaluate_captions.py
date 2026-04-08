import os
import argparse
import time
from transformers import BertForSequenceClassification, BertTokenizer, AutoTokenizer, AutoModel
from bert_score import score
import json
import torch
import torch.nn.functional as F

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


def get_batch_dict_num_score(generated_captions, gt_metadata, stat_inf_judge_model):
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
        #print("------------------------------------\nGT: ", gt_metadata)
        #print("Gen: ", gen_capt)
        try:
            num_dict_gen = extract_num_dict_from_text(gen_capt, model=stat_inf_judge_model)
            #print("\nGen dict: ", num_dict_gen)
            num_dict_gt = extract_num_dict_from_dict(gt_metadata, model=stat_inf_judge_model)
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
            print(e)
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


def get_batch_simcse_score(generated_captions, gt_captions, model, tokenizer, device):
    """
    Compute SimCSE similarity scores between generated and ground truth captions.
    Returns the average cosine similarity across all caption pairs.
    """
    similarity_sum = 0

    for gen_capt, gt_capt in zip(generated_captions, gt_captions):
        texts = [gt_capt, gen_capt]
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)

        with torch.no_grad():
            embeddings = model(**inputs).pooler_output
            embeddings = F.normalize(embeddings, p=2, dim=1)
            similarity = F.cosine_similarity(embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0)).item()

        similarity_sum += similarity

    return similarity_sum / len(generated_captions)


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Evaluate generated captions against ground truth')
    parser.add_argument('--generated_captions_folder_path', type=str, default=None,
                        help='Path to folder containing generated captions (overrides config file)')
    parser.add_argument('--gt_captions_folder_path', type=str, default=None,
                        help='Path to folder containing ground truth captions (overrides config file)')
    parser.add_argument('--gt_metadata_folder_path', type=str, default=None,
                        help='Path to folder containing ground truth metadata (overrides config file)')
    parser.add_argument('--evaluation_results_folder_path', type=str, default=None,
                        help='Path to folder for saving evaluation results (overrides config file)')
    args = parser.parse_args()

    config = load_config()

    # Debug: Print command-line arguments
    print("\n=== Command-line Arguments ===")
    print(f"generated_captions_folder_path argument: {args.generated_captions_folder_path}")
    print(f"Config file value BEFORE override: {config['path']['generated_captions_folder_path']}")

    # Override config with command-line arguments if provided
    if args.generated_captions_folder_path:
        config['path']['generated_captions_folder_path'] = args.generated_captions_folder_path
    if args.gt_captions_folder_path:
        config['path']['gt_captions_folder_path'] = args.gt_captions_folder_path
    if args.gt_metadata_folder_path:
        config['path']['gt_metadata_folder_path'] = args.gt_metadata_folder_path
    if args.evaluation_results_folder_path:
        config['path']['evaluation_results_folder_path'] = args.evaluation_results_folder_path

    print(f"Config file value AFTER override: {config['path']['generated_captions_folder_path']}")
    print("==============================\n")

    eval_model = config['path']['generated_captions_folder_path'].split("/")[-1].replace(" ", "_")
    #eval_model = "gemini-2.0-flash_text" # _text

    # Load the extraction model from config
    stat_inf_judge_model = config['model']['extraction_model']

    print("\nEvaluating captions from: ", config['path']['generated_captions_folder_path'])
    print("Ground truth captions from: ", config['path']['gt_captions_folder_path'])
    print(f"Using {stat_inf_judge_model} for statistical inference judgement")

    # Initialize SimCSE model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    simcse_model_name = "princeton-nlp/sup-simcse-roberta-large"
    print(f"Loading SimCSE model: {simcse_model_name}")
    simcse_tokenizer = AutoTokenizer.from_pretrained(simcse_model_name)
    simcse_model = AutoModel.from_pretrained(simcse_model_name).eval().to(device)
    print(f"SimCSE model loaded on device: {device}")

    generated_captions_folder_path = config['path']['generated_captions_folder_path']
    
    
    gt_captions_folder_path = config['path']['gt_captions_folder_path']
    
    for filename in os.listdir(gt_captions_folder_path):
        if not filename.endswith(".txt"):
            continue
        if "test" not in filename:
            new_filename = filename.replace(".txt", "_test.txt")
            os.rename(
                os.path.join(gt_captions_folder_path, filename),
                os.path.join(gt_captions_folder_path, new_filename)
            )
            
    for filename in os.listdir(generated_captions_folder_path):
        if not filename.endswith(".txt"):
            continue
        if "test" not in filename:
            new_filename = filename.replace(".txt", "_test.txt")
            os.rename(
                os.path.join(generated_captions_folder_path, filename),
                os.path.join(generated_captions_folder_path, new_filename)
            )
    #generated_captions_folder_path = f"/home/ubuntu/thesis/data/samples/len 10/generated captions/{eval_model}"
    generated_caption_paths = [os.path.join(generated_captions_folder_path, filename) for filename in os.listdir(generated_captions_folder_path)]
    generated_caption_paths.sort()
    
    #keep_domains = ['agriculture', 'crime', 'demography', 'walmart'] ############### To ignore some domains
    #generated_caption_paths = [path for path in generated_caption_paths if path.split("/")[-1].split("_")[0] in keep_domains]

    gt_captions_folder_path = config['path']['gt_captions_folder_path']
    #gt_captions_folder_path = "/home/ubuntu/thesis/data/samples/len 10/captions"
    gt_caption_paths = [os.path.join(gt_captions_folder_path, filename) for filename in os.listdir(gt_captions_folder_path)]
    gt_caption_paths.sort()
    
    gt_metadata_folder_path = config['path']['gt_metadata_folder_path']
    gt_metadata_paths = [os.path.join(gt_metadata_folder_path, filename) for filename in os.listdir(gt_metadata_folder_path)]
    gt_metadata_paths.sort()


    # Filter to intersection of files that exist in both folders
    gt_filenames = {os.path.basename(path) for path in gt_caption_paths}
    generated_filenames = {os.path.basename(path) for path in generated_caption_paths}
    common_filenames = gt_filenames & generated_filenames

    gt_caption_paths = [path for path in gt_caption_paths if os.path.basename(path) in common_filenames]
    gt_caption_paths.sort()

    generated_caption_paths = [path for path in generated_caption_paths if os.path.basename(path) in common_filenames]
    generated_caption_paths.sort()

    gt_metadata_paths = [path for path in gt_metadata_paths if os.path.splitext(os.path.basename(path))[0]+".txt" in common_filenames]
    gt_metadata_paths.sort()
    
    print(f"GT files: {len(gt_caption_paths)}, generated files: {len(generated_caption_paths)}")
    #print(gt_metadata_paths)
    #assert len(generated_caption_paths) == len(gt_caption_paths)
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
    
    print(gt_metadatas.keys())

    
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
            
    for i, dataset in enumerate(dataset_gt_caption_paths.keys()):
        #if dataset in ['demography', 'agriculture', 'co2', 'diet','walmart','road injuries', 'online retail']: 
           # continue
    #for dataset in ["demography"]:
        if result_dict[dataset] != {}:
            continue
        
        gen_capts = generated_captions[dataset]
        gt_capts = gt_captions[dataset]
        gt_metadata = gt_metadatas[dataset]
    
        print(f"\n\n{i+1}/{len(list(dataset_gt_caption_paths.keys()))}: {dataset}: {len(gen_capts)} captions are being scored...")
        
        
        ################################# Numeric Score 2.0 ###############################################

        num_score_dict = get_batch_dict_num_score(generated_captions=gen_capts, gt_metadata=gt_metadata, stat_inf_judge_model=stat_inf_judge_model)

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

        print(f"METEOR SCORE: {round(meteor, 3)}")

        result_dict[dataset]['meteor score'] = round(meteor, 3)

        ################################# SIMCSE SCORE ###############################################

        simcse = get_batch_simcse_score(generated_captions=gen_capts, gt_captions=gt_capts,
                                        model=simcse_model, tokenizer=simcse_tokenizer, device=device)

        print(f"SIMCSE SCORE: {round(simcse, 3)}")

        result_dict[dataset]['simcse score'] = round(simcse, 3)


        ################################# ORACLE SCORE ###############################################
        
        """oracle_sc = get_batch_score(generated_captions=gen_capts, gt_captions=gt_capts, score_function=oracle_score)
        oracle_sc = round(oracle_sc/100, 3)
        print(f"ORACLE SCORE: {round(oracle_sc, 3)}")
        
        result_dict[dataset]['oracle score'] = round(oracle_sc, 3)"""
        
        
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
        "simcse score": 0,
        #"oracle score": 0
    }

    # Don't include 'average' in the dataset count or loop
    datasets = [d for d in result_dict if d != "average"]
    
    #datasets = [d for d in result_dict if d not in ['demography', 'agriculture', 'co2', 'diet','walmart','road injuries', 'online retail', 'average']]
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
        average_scores["simcse score"] += result_dict[dataset]["simcse score"]
        #average_scores["oracle score"] += result_dict[dataset]["oracle score"]

    # Compute the mean for each score
    average_scores["bert score"]["precision"] /= dataset_count
    average_scores["bert score"]["recall"] /= dataset_count
    average_scores["bert score"]["f1"] /= dataset_count
    average_scores["numeric score 2.0"]["minimum"] /= dataset_count
    average_scores["numeric score 2.0"]["maximum"] /= dataset_count
    average_scores["numeric score 2.0"]["mean"] /= dataset_count-mean_nones
    if dataset_count-std_nones != 0:
        average_scores["numeric score 2.0"]["std"] /= dataset_count-std_nones
    else:
        average_scores["numeric score 2.0"]["std"] = None
    average_scores["numeric score"] /= dataset_count
    average_scores["bleu score"] /= dataset_count
    average_scores["rouge score"] /= dataset_count
    average_scores["meteor score"] /= dataset_count
    average_scores["simcse score"] /= dataset_count
    #average_scores["oracle score"] /= dataset_count

    # Now safe to modify the dictionary
    result_dict["average"] = average_scores

    # Save the result
    save_file(result_dict, filepath=save_path)
    print("\n\nDone for ", eval_model)
    print("Saved to ", save_path)
            
        
    
if __name__ == "__main__":
    main()
    """gen = ["The time series shows a fluctuating pattern over the four-year period. Starting at 116 severe injuries in 2016, there was a significant drop to 68 in 2017, followed by a gradual increase to 81 in 2018 and a slight rise to 82 in 2019. On average, there were approximately 86.75 severe injuries per year. Compared to a hypothetical global average of 100 severe injuries per year in similar regions, Shasta County's average of 86.75 is slightly lower, indicating a minor difference in severity rates."]
    gt_meta = [{
        "average time series of this type of location": [
            215.61,
            210.18,
            219.47,
            213.53
        ],
        "end year": 2019,
        "geotype": "County",
        "location": "Shasta",
        "maximum of this specific series": 116.0,
        "mean of this specific series": 86.75,
        "minimum of this specific series": 68.0,
        "mode": "All modes",
        "sampling frequency": "yearly",
        "severity": "Severe Injury",
        "standard deviation of this specific series": 17.77,
        "standard deviation of this type of location": 3.37,
        "starting year": 2016,
        "total population": 169543
        }]
    
    get_batch_dict_num_score(generated_captions=gen, gt_metadata=gt_meta)"""