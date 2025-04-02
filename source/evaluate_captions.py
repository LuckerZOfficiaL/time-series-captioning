import os
from transformers import BertForSequenceClassification, BertTokenizer
from bert_score import score

from helpers import(
    load_config,
)



def main():
    config = load_config()

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
    #generated_caption_paths = generated_caption_paths[:5]
    #gt_caption_paths = gt_caption_paths[:5]


    # Read the captions into lists of strings
    generated_captions = []
    for generated_caption_path in generated_caption_paths:
        with open(generated_caption_path, 'r') as file:
            generated_caption = file.read()
            generated_captions.append(generated_caption)

    gt_captions = []
    for gt_caption_path in gt_caption_paths:
        with open(gt_caption_path, 'r') as file:
            gt_caption = file.read()
            gt_captions.append(gt_caption)

    print(f"\n{len(generated_captions)} captions are being scored...")
    
    """
        P (Precision): Measures how much of the candidate text's meaning is captured in the reference text.
        R (Recall): Measures how much of the reference text's meaning is captured in the candidate text.
        F1 (F1-score): The harmonic mean of precision and recall, providing a balanced similarity measure.
    """
    P, R, F1 = score(generated_captions, gt_captions, lang="en", model_type=config['eval']['bertscore_model'])

    # Print the BERT scores for all
    """for i in range(len(F1)):
        print(f"\n{generated_caption_paths[i].split("/")[-1][:-4]}")
        print(f"P: {round(P[i], 3)}, R: {round(R[i], 3)}, F1: {round(F1[i], 3)}")"""
    
    # Print the average scores across samples
    p_mean = sum(P) / len(P)
    r_mean = sum(R) / len(R)
    f1_mean = sum(F1) / len(F1)
    print(f"Mean P: {round(p_mean.item(), 3)}, Mean R: {round(r_mean.item(), 3)}, Mean F1: {round(f1_mean.item(), 3)}")


if __name__ == "__main__":
    main()