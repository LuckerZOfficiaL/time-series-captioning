import json
import os
import re

GROUND_TRUTH_DIR = "caption_retrieval_easy/ground_truth"
ANSWER_DIR = "phi_etiology_test_with_image"

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
        import pdb; pdb.set_trace()

    answer = data.get("answer", "").strip()
    m = re.match(r'^\(?\s*([A-Za-z])\s*\)?', answer)
    return m.group(1).upper() if m else None

def get_answer(a):
    options = ['A', 'B', 'C', 'D']
    answers = [x for x in options if x in a]
    if len(answers) != 1:
        raise ValueError(a)
    return answers[0]

def eval_score(answer_dir, ground_truth_dir):
    answers = []
    ground_truths = []
    for ts_name in os.listdir(answer_dir):
        with open(os.path.join(answer_dir, ts_name)) as fh:
            answer = extract_choice(fh.read())
            assert answer in {'A', 'B', 'C', 'D'}
            answers.append(answer)
        with open(os.path.join(ground_truth_dir, ts_name)) as fh:
            gt = fh.read()
            assert gt in {'A', 'B', 'C', 'D'}
            ground_truths.append(gt)
    accuracy_rate = len([i for i, g in enumerate(ground_truths) if g == answers[i]]) / len(answers)
    print(f"Answer dir: {answer_dir}")
    print(f"Ground truth dir: {ground_truth_dir}")
    print(f"Accuracy rate: {accuracy_rate:.3f}")
    print("---------------------")

if __name__ == "__main__":
    eval_score(ANSWER_DIR, GROUND_TRUTH_DIR)
