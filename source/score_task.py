import json
import os
import re

TASK_DIR = "data/samples/new samples no overlap/tasks/caption_retrieval_cross_domain"
ANSWER_DIR = "qwen_inference_results/caption_retrieval_cross_domain"


def load_ground_truth(task_dir):
    with open(os.path.join(task_dir, "tasks.json")) as fh:
        tasks = json.load(fh)
    return {t["ts_name"]: t["ground_truth"] for t in tasks}

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

def eval_score(answer_dir, task_dir):
    answers = {}
    ground_truths = load_ground_truth(task_dir)
    assert all(v in {'A', 'B', 'C', 'D'} for v in ground_truths.values())
    for ts_file in os.listdir(answer_dir):
        ts_name = ts_file.replace(".txt", "")
        with open(os.path.join(answer_dir, ts_file)) as fh:
            answer = extract_choice(fh.read())
            answers[ts_name] = answer
    accuracy_rate = len([k for k, v in answers.items() if v == ground_truths[k]]) / len(answers)
    print(f"Answer dir: {answer_dir}")
    print(f"Ground truth dir: {task_dir}")
    print(f"Num answers: {len(answers)}")
    print(f"Accuracy rate: {accuracy_rate:.3f}")
    print("---------------------")

if __name__ == "__main__":
    eval_score(ANSWER_DIR, TASK_DIR)
