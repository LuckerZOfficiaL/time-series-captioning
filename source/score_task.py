import json
import os
import re

GROUND_TRUTH_DIR = "data/samples/new samples no overlap/tasks/caption_retrieval_cross_domain_with_image/ground_truth"
ANSWER_DIR = "llava_caption_retrieval_with_image_easy"

def get_caption_retrieval_prompts(data_dir):
    prompt_dir = os.path.join(data_dir, 'prompts')
    all_prompts = []
    for p in os.listdir(prompt_dir):
        with open(os.path.join(prompt_dir, p)) as fh:
            all_prompts.append(json.load(fh))
    return all_prompts

def extract_choice(json_str):
    json_str = json_str.replace('```', '').replace('json', '')
    json_str = re.sub(r'("answer"\s*:\s*)([A-Za-z])', r'\1"\2"', json_str)
    try:
        data = json.loads(json_str)
    except:
        print("INVALID ANSWER:")
        print(json_str)
        return ""
    
    answer = data.get("answer", "").strip()
    m = re.match(r'^\(?\s*([A-Za-z])\s*\)?', answer)
    return m.group(1).upper() if m else None

def eval_score(answer_dir, ground_truth_dir):
    answers = []
    ground_truths = []
    for ts_name in os.listdir(answer_dir):
        with open(os.path.join(answer_dir, ts_name)) as fh:
            answer = extract_choice(fh.read())
            answers.append(answer)
        with open(os.path.join(ground_truth_dir, ts_name)) as fh:
            gt = fh.read()
            assert gt in {'A', 'B', 'C', 'D'}
            ground_truths.append(gt)
    accuracy_rate = len([i for i, g in enumerate(ground_truths) if g == answers[i]]) / len(answers)
    print(f"Answer dir: {answer_dir}")
    print(f"Ground truth dir: {ground_truth_dir}")
    print(f"Num answers: {len(answers)}")
    print(f"Accuracy rate: {accuracy_rate:.3f}")
    print("---------------------")

if __name__ == "__main__":
    eval_score(ANSWER_DIR, GROUND_TRUTH_DIR)
