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


def get_batch_score(generated_captions, gt_captions, score_function):
    score_sum = 0
    for gen_capt, gt_capt in zip(generated_captions, gt_captions):
        score_sum += score_function(gen_capt, gt_capt)
    return score_sum / len(generated_captions)


def calculate_metrics_for_batch(generated_captions, gt_captions, config):
    """Calculate all metrics for a batch of captions"""
    metrics = {}
    
    # BERT SCORE
    P, R, F1 = score(generated_captions, gt_captions, lang="en", model_type=config['eval']['bertscore_model'])
    p_mean = sum(P) / len(P)
    r_mean = sum(R) / len(R)
    f1_mean = sum(F1) / len(F1)
    
    metrics['bert_score'] = {
        "precision": round(p_mean.item(), 3),
        "recall": round(r_mean.item(), 3),
        "f1": round(f1_mean.item(), 3)
    }
    
    # BLEU SCORE
    bleu = get_batch_score(generated_captions=generated_captions, gt_captions=gt_captions, score_function=bleu_score)
    metrics['bleu_score'] = round(bleu, 3)
    
    # ROUGE SCORE
    rouge = get_batch_score(generated_captions=generated_captions, gt_captions=gt_captions, score_function=rouge_score)
    metrics['rouge_score'] = round(rouge, 3)
    
    # METEOR SCORE
    meteor = get_batch_score(generated_captions=generated_captions, gt_captions=gt_captions, score_function=meteor_score)
    metrics['meteor_score'] = round(meteor, 3)
    
    # ORACLE SCORE
    oracle_sc = get_batch_score(generated_captions=generated_captions, gt_captions=gt_captions, score_function=oracle_score)
    oracle_sc = round(oracle_sc/100, 3)
    metrics['oracle_score'] = round(oracle_sc, 3)
    
    return metrics


def update_aggregate_metrics(result_dict, batch_metrics, batch_size, total_processed):
    """Update aggregate metrics with new batch results"""
    
    if 'aggregate_metrics' not in result_dict:
        result_dict['aggregate_metrics'] = {}
    
    aggregate = result_dict['aggregate_metrics']
    
    # Initialize if first batch
    if total_processed == batch_size:
        aggregate['bert score'] = batch_metrics['bert_score'].copy()
        aggregate['bleu score'] = batch_metrics['bleu_score']
        aggregate['rouge score'] = batch_metrics['rouge_score']
        aggregate['meteor score'] = batch_metrics['meteor_score']
        aggregate['oracle score'] = batch_metrics['oracle_score']
    else:
        # Update running averages
        prev_count = total_processed - batch_size
        new_weight = batch_size / total_processed
        prev_weight = prev_count / total_processed
        
        # Update BERT score components
        aggregate['bert score']['precision'] = round(
            prev_weight * aggregate['bert score']['precision'] + 
            new_weight * batch_metrics['bert_score']['precision'], 3
        )
        aggregate['bert score']['recall'] = round(
            prev_weight * aggregate['bert score']['recall'] + 
            new_weight * batch_metrics['bert_score']['recall'], 3
        )
        aggregate['bert score']['f1'] = round(
            prev_weight * aggregate['bert score']['f1'] + 
            new_weight * batch_metrics['bert_score']['f1'], 3
        )
        
        # Update other metrics
        aggregate['bleu score'] = round(
            prev_weight * aggregate['bleu score'] + 
            new_weight * batch_metrics['bleu_score'], 3
        )
        aggregate['rouge score'] = round(
            prev_weight * aggregate['rouge score'] + 
            new_weight * batch_metrics['rouge_score'], 3
        )
        aggregate['meteor score'] = round(
            prev_weight * aggregate['meteor score'] + 
            new_weight * batch_metrics['meteor_score'], 3
        )
        aggregate['oracle score'] = round(
            prev_weight * aggregate['oracle score'] + 
            new_weight * batch_metrics['oracle_score'], 3
        )


def main():
    config = load_config()
    eval_model = config['path']['generated_captions_folder_path'].split("/")[-1].replace(" ", "_")
    
    print("\nEvaluating captions from: ", eval_model)
    print("\nGround Truths are in: ", config['path']['gt_captions_folder_path'])

    generated_captions_folder_path = config['path']['generated_captions_folder_path']
    generated_caption_paths = [os.path.join(generated_captions_folder_path, filename) 
                              for filename in os.listdir(generated_captions_folder_path) 
                              if filename.endswith(".txt")]
    generated_caption_paths.sort()

    gt_captions_folder_path = config['path']['gt_captions_folder_path']
    gt_caption_paths = [os.path.join(gt_captions_folder_path, filename) 
                       for filename in os.listdir(gt_captions_folder_path) 
                       if filename.endswith(".txt")]
    gt_caption_paths.sort()

    print("len(generated_caption_paths), len(gt_caption_paths): ", len(generated_caption_paths), len(gt_caption_paths))
    
    if len(generated_caption_paths) < len(gt_caption_paths):
        # Filter gt_caption_paths to include only those that are also in generated_caption_paths
        generated_filenames = {os.path.basename(path) for path in generated_caption_paths}
        gt_caption_paths = [path for path in gt_caption_paths if os.path.basename(path) in generated_filenames]
        gt_caption_paths.sort()

    # Ensure we have matching pairs
    assert len(generated_caption_paths) == len(gt_caption_paths), \
        f"Mismatch in number of files: {len(generated_caption_paths)} vs {len(gt_caption_paths)}"
    
    save_path = config['path']['evaluation_results_folder_path'] + "/" + eval_model + ".json"

    # Load existing results if they exist
    if os.path.exists(save_path):
        print(f"Loading existing results from {save_path}...")
        with open(save_path, 'r') as file:
            result_dict = json.load(file)
        
        processed_files = set(result_dict.get('processed_files', []))
        print(f"Found {len(processed_files)} already processed files")
    else:   
        print(f"Creating result dictionary from scratch...")
        result_dict = {
            'processed_files': [],
            'individual_results': {},
            'aggregate_metrics': {},
            'total_files': len(generated_caption_paths)
        }
        processed_files = set()

    # Filter out already processed files
    remaining_pairs = []
    for gen_path, gt_path in zip(generated_caption_paths, gt_caption_paths):
        filename = os.path.basename(gen_path)
        if filename not in processed_files:
            remaining_pairs.append((gen_path, gt_path, filename))

    if not remaining_pairs:
        print("All files have already been processed!")
        print(f"Final aggregate results:")
        if 'aggregate_metrics' in result_dict:
            for metric, value in result_dict['aggregate_metrics'].items():
                print(f"{metric.upper()}: {value}")
        return

    print(f"\n{len(remaining_pairs)} files remaining to process...")

    # Process files in batches (you can adjust batch size)
    batch_size = min(50, len(remaining_pairs))  # Process in batches of 50 or less
    
    for i in range(0, len(remaining_pairs), batch_size):
        batch_pairs = remaining_pairs[i:i+batch_size]
        
        print(f"\nProcessing batch {i//batch_size + 1}/{(len(remaining_pairs)-1)//batch_size + 1} "
              f"({len(batch_pairs)} files)...")
        
        # Read captions for this batch
        batch_generated_captions = []
        batch_gt_captions = []
        batch_filenames = []
        
        for gen_path, gt_path, filename in batch_pairs:
            with open(gen_path, 'r') as file:
                batch_generated_captions.append(file.read())
            with open(gt_path, 'r') as file:
                batch_gt_captions.append(file.read())
            batch_filenames.append(filename)

        # Calculate metrics for this batch
        batch_metrics = calculate_metrics_for_batch(batch_generated_captions, batch_gt_captions, config)
        
        # Store individual results
        for j, filename in enumerate(batch_filenames):
            result_dict['individual_results'][filename] = {
                'bert_score': {
                    'precision': batch_metrics['bert_score']['precision'],
                    'recall': batch_metrics['bert_score']['recall'], 
                    'f1': batch_metrics['bert_score']['f1']
                },
                'bleu_score': batch_metrics['bleu_score'],
                'rouge_score': batch_metrics['rouge_score'],
                'meteor_score': batch_metrics['meteor_score'],
                'oracle_score': batch_metrics['oracle_score']
            }
            result_dict['processed_files'].append(filename)

        # Update aggregate metrics
        total_processed = len(result_dict['processed_files'])
        update_aggregate_metrics(result_dict, batch_metrics, len(batch_pairs), total_processed)
        
        # Print current batch results
        print(f"Batch metrics:")
        print(f"  BERT SCORE - P: {batch_metrics['bert_score']['precision']}, "
              f"R: {batch_metrics['bert_score']['recall']}, F1: {batch_metrics['bert_score']['f1']}")
        print(f"  BLEU: {batch_metrics['bleu_score']}")
        print(f"  ROUGE: {batch_metrics['rouge_score']}")
        print(f"  METEOR: {batch_metrics['meteor_score']}")
        print(f"  ORACLE: {batch_metrics['oracle_score']}")
        
        print(f"\nRunning aggregate metrics (after {total_processed} files):")
        agg = result_dict['aggregate_metrics']
        print(f"  BERT SCORE - P: {agg['bert score']['precision']}, "
              f"R: {agg['bert score']['recall']}, F1: {agg['bert score']['f1']}")
        print(f"  BLEU: {agg['bleu score']}")
        print(f"  ROUGE: {agg['rouge score']}")
        print(f"  METEOR: {agg['meteor score']}")
        print(f"  ORACLE: {agg['oracle score']}")

        # Save intermediate results
        save_file(result_dict, filepath=save_path)
        print(f"Saved intermediate results ({total_processed}/{result_dict['total_files']} files processed)")

    # Copy aggregate metrics to the old format for backward compatibility
    if 'aggregate_metrics' in result_dict:
        result_dict['bert score'] = result_dict['aggregate_metrics']['bert score']
        result_dict['bleu score'] = result_dict['aggregate_metrics']['bleu score']
        result_dict['rouge score'] = result_dict['aggregate_metrics']['rouge score']
        result_dict['meteor score'] = result_dict['aggregate_metrics']['meteor score']
        result_dict['oracle score'] = result_dict['aggregate_metrics']['oracle score']

    # Final save
    save_file(result_dict, filepath=save_path)
    
    print(f"\nFinal results for {eval_model}:")
    print(f"BERT SCORE - P: {result_dict['bert score']['precision']}, "
          f"R: {result_dict['bert score']['recall']}, F1: {result_dict['bert score']['f1']}")
    print(f"BLEU SCORE: {result_dict['bleu score']}")
    print(f"ROUGE SCORE: {result_dict['rouge score']}")
    print(f"METEOR SCORE: {result_dict['meteor score']}")
    print(f"ORACLE SCORE: {result_dict['oracle score']}")
    

if __name__ == "__main__":
    main()