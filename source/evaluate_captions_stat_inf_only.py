import os
import argparse
import json

from helpers import(
    load_config,
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
        "minimum_attempts": 0,
        "maximum_attempts": 0,
        "mean_attempts": 0,
        "std_attempts": 0,
    }

    failed_extractions = 0
    total_captions = len(generated_captions)

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

            # Track attempts: if num_dict_gen has a non-null value, the model attempted to report it
            if num_dict_gen.get('minimum') is not None:
                result_dict['minimum_attempts'] += 1
            if num_dict_gen.get('maximum') is not None:
                result_dict['maximum_attempts'] += 1
            if num_dict_gen.get('mean') is not None:
                result_dict['mean_attempts'] += 1
            if num_dict_gen.get('std') is not None:
                result_dict['std_attempts'] += 1

            # Track accuracy (only for attempted statistics)
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

    # Calculate accuracy (correct detections / attempted detections)
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

    # Calculate attempt rates (attempts / total captions)
    result_dict['minimum_attempt_rate'] = result_dict['minimum_attempts'] / total_captions if total_captions > 0 else 0
    result_dict['maximum_attempt_rate'] = result_dict['maximum_attempts'] / total_captions if total_captions > 0 else 0
    result_dict['mean_attempt_rate'] = result_dict['mean_attempts'] / total_captions if total_captions > 0 else 0
    result_dict['std_attempt_rate'] = result_dict['std_attempts'] / total_captions if total_captions > 0 else 0

    # Calculate success rate normalized by attempt rate (correct detections / total captions)
    result_dict['minimum_success_rate'] = (result_dict['minimum'] * (len(generated_captions) - result_dict["minimum_nulls"])) / total_captions if total_captions > 0 else 0
    result_dict['maximum_success_rate'] = (result_dict['maximum'] * (len(generated_captions) - result_dict["maximum_nulls"])) / total_captions if total_captions > 0 else 0

    if result_dict['mean'] is not None:
        result_dict['mean_success_rate'] = (result_dict['mean'] * (len(generated_captions) - result_dict["mean_nulls"])) / total_captions if total_captions > 0 else 0
    else:
        result_dict['mean_success_rate'] = 0

    if result_dict['std'] is not None:
        result_dict['std_success_rate'] = (result_dict['std'] * (len(generated_captions) - result_dict["std_nulls"])) / total_captions if total_captions > 0 else 0
    else:
        result_dict['std_success_rate'] = 0

    #del result_dict['minimum_nulls']
    #del result_dict['maximum_nulls']
    #del result_dict['mean_nulls']
    #del result_dict['std_nulls']

    return result_dict
    
    



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
    print("Ground truth metadata from: ", config['path']['gt_metadata_folder_path'])
    print(f"Using {stat_inf_judge_model} for statistical inference judgement")

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
    #for dataset in ["demography"]:
        if result_dict[dataset] != {}:
            continue

        gen_capts = generated_captions[dataset]
        gt_metadata = gt_metadatas[dataset]

        print(f"\n\n{i+1}/{len(list(dataset_gt_caption_paths.keys()))}: {dataset}: {len(gen_capts)} captions are being scored...")


        ################################# Numeric Score 2.0 ###############################################

        num_score_dict = get_batch_dict_num_score(generated_captions=gen_capts, gt_metadata=gt_metadata, stat_inf_judge_model=stat_inf_judge_model)

        print(f"NUMERIC SCORE 2.0: {num_score_dict}")

        result_dict[dataset]['numeric score 2.0'] = num_score_dict


        ################################ SAVE CHECKPOINT ###############################################
        save_file(result_dict, filepath=save_path)
        
        
    ############################### AVERAGE SCORE #############################################

    average_scores = {
        "numeric score 2.0":{
            "minimum": 0,
            "maximum": 0,
            "mean": 0,
            "std": 0,
            "minimum_attempts": 0,
            "maximum_attempts": 0,
            "mean_attempts": 0,
            "std_attempts": 0,
            "minimum_attempt_rate": 0,
            "maximum_attempt_rate": 0,
            "mean_attempt_rate": 0,
            "std_attempt_rate": 0,
            "minimum_success_rate": 0,
            "maximum_success_rate": 0,
            "mean_success_rate": 0,
            "std_success_rate": 0,
        }
    }

    # Don't include 'average' in the dataset count or loop
    datasets = [d for d in result_dict if d != "average"]
    dataset_count = len(datasets)
    mean_nones = 0
    std_nones = 0
    mean_success_rate_nones = 0
    std_success_rate_nones = 0

    for dataset in datasets:
        average_scores["numeric score 2.0"]["minimum"] += result_dict[dataset]["numeric score 2.0"]["minimum"]
        average_scores["numeric score 2.0"]["maximum"] += result_dict[dataset]["numeric score 2.0"]["maximum"]

        # Attempt counts
        average_scores["numeric score 2.0"]["minimum_attempts"] += result_dict[dataset]["numeric score 2.0"]["minimum_attempts"]
        average_scores["numeric score 2.0"]["maximum_attempts"] += result_dict[dataset]["numeric score 2.0"]["maximum_attempts"]
        average_scores["numeric score 2.0"]["mean_attempts"] += result_dict[dataset]["numeric score 2.0"]["mean_attempts"]
        average_scores["numeric score 2.0"]["std_attempts"] += result_dict[dataset]["numeric score 2.0"]["std_attempts"]

        # Attempt rates
        average_scores["numeric score 2.0"]["minimum_attempt_rate"] += result_dict[dataset]["numeric score 2.0"]["minimum_attempt_rate"]
        average_scores["numeric score 2.0"]["maximum_attempt_rate"] += result_dict[dataset]["numeric score 2.0"]["maximum_attempt_rate"]
        average_scores["numeric score 2.0"]["mean_attempt_rate"] += result_dict[dataset]["numeric score 2.0"]["mean_attempt_rate"]
        average_scores["numeric score 2.0"]["std_attempt_rate"] += result_dict[dataset]["numeric score 2.0"]["std_attempt_rate"]

        # Success rates
        average_scores["numeric score 2.0"]["minimum_success_rate"] += result_dict[dataset]["numeric score 2.0"]["minimum_success_rate"]
        average_scores["numeric score 2.0"]["maximum_success_rate"] += result_dict[dataset]["numeric score 2.0"]["maximum_success_rate"]

        if result_dict[dataset]["numeric score 2.0"]["mean"] != None:
            average_scores["numeric score 2.0"]["mean"] += result_dict[dataset]["numeric score 2.0"]["mean"]
        else:
            mean_nones += 1

        if result_dict[dataset]["numeric score 2.0"]["std"] != None:
            average_scores["numeric score 2.0"]["std"] += result_dict[dataset]["numeric score 2.0"]["std"]
        else:
            std_nones += 1

        # Handle success rate for mean and std (can be 0 if the accuracy was None)
        if result_dict[dataset]["numeric score 2.0"]["mean_success_rate"] != 0 or result_dict[dataset]["numeric score 2.0"]["mean"] is not None:
            average_scores["numeric score 2.0"]["mean_success_rate"] += result_dict[dataset]["numeric score 2.0"]["mean_success_rate"]
        else:
            mean_success_rate_nones += 1

        if result_dict[dataset]["numeric score 2.0"]["std_success_rate"] != 0 or result_dict[dataset]["numeric score 2.0"]["std"] is not None:
            average_scores["numeric score 2.0"]["std_success_rate"] += result_dict[dataset]["numeric score 2.0"]["std_success_rate"]
        else:
            std_success_rate_nones += 1

    # Compute the mean for each score
    average_scores["numeric score 2.0"]["minimum"] /= dataset_count
    average_scores["numeric score 2.0"]["maximum"] /= dataset_count
    average_scores["numeric score 2.0"]["mean"] /= dataset_count-mean_nones
    if dataset_count-std_nones != 0:
        average_scores["numeric score 2.0"]["std"] /= dataset_count-std_nones
    else:
        average_scores["numeric score 2.0"]["std"] = None

    # Average the attempt counts and rates
    average_scores["numeric score 2.0"]["minimum_attempts"] /= dataset_count
    average_scores["numeric score 2.0"]["maximum_attempts"] /= dataset_count
    average_scores["numeric score 2.0"]["mean_attempts"] /= dataset_count
    average_scores["numeric score 2.0"]["std_attempts"] /= dataset_count

    average_scores["numeric score 2.0"]["minimum_attempt_rate"] /= dataset_count
    average_scores["numeric score 2.0"]["maximum_attempt_rate"] /= dataset_count
    average_scores["numeric score 2.0"]["mean_attempt_rate"] /= dataset_count
    average_scores["numeric score 2.0"]["std_attempt_rate"] /= dataset_count

    average_scores["numeric score 2.0"]["minimum_success_rate"] /= dataset_count
    average_scores["numeric score 2.0"]["maximum_success_rate"] /= dataset_count
    average_scores["numeric score 2.0"]["mean_success_rate"] /= (dataset_count - mean_success_rate_nones) if (dataset_count - mean_success_rate_nones) > 0 else 1
    average_scores["numeric score 2.0"]["std_success_rate"] /= (dataset_count - std_success_rate_nones) if (dataset_count - std_success_rate_nones) > 0 else 1

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