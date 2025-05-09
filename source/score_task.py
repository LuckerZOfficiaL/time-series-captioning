from collections import defaultdict
import json
import os
from pathlib import Path
import re
import sys

TASK_DIR = "data/samples/new samples no overlap/hard_questions_small"
ANSWER_DIR = "{model}_inference_results_small"
_ALLOWED_ANSWERS = {"a", "b", "c", "d", "true", "false"}


def load_ground_truth(task_dir):
    with open(os.path.join(task_dir, "tasks.json")) as fh:
        tasks = json.load(fh)
    return {t["ts_name"]: t["ground_truth"] for t in tasks}

def _parse_start(s: str):
    # Grab the first contiguous token after leading whitespace
    m = re.match(r"\s*(\S+)", s)
    if not m:
        print("String is empty or whitespace only.")
        return None
    token = m.group(1).lower()
    if token not in _ALLOWED_ANSWERS:
        print(f"Unrecognised answer '{m.group(1)}' in: {s!r}")
        return None
    return token.title()
        

def _attempt_parse_str(s):
    s = s.lower()
    assert not (("false" in s) and ("true" in s))
    if "false" in s:
        return 'False'
    if "true" in s:
        return 'True'
    # Lots of hardcoded reformats necessary from different LLM answers
    str_reformat = s.replace('.', '').replace('"', '').replace("'", '')
    str_reformat = s.replace(")", "").replace("(", "").replace("answer:", "").split(' ')
    for x in str_reformat:
        if x in _ALLOWED_ANSWERS:
            return _parse_start(x)
    print(f"INVALID ANSWER:\n {s}")

    return ""

def extract_choice(json_str):
    orig_str = json_str
    json_str = json_str.replace('```', '').replace('json', '').replace('.', '')
    if "{" in json_str and "}" in json_str:
        json_str = json_str.split('}')[0] + '}'
    json_str = re.sub(r'("answer"\s*:\s*)([A-Za-z])', r'\1"\2"', json_str)
    try:
        data = json.loads(json_str)
    except:
        return _attempt_parse_str(json_str)
    
    try:
        answer = data.get("answer", "").strip()
    except AttributeError:
        # handle LLM answering only 'true' or 'false'
        if type(data) == bool:
            return str(data)
        return _attempt_parse_str(data)

    return _parse_start(answer)


def score_breakdown(answers, ground_truths):
    scores = {k: (v == ground_truths[k]) for k,v in answers.items()}
    # TODO: make this a global constant somewhere and import it
    datasets = ["air quality", "crime", "border crossing", "demography", "road injuries", 
                "covid", "co2", "diet", "online retail", "walmart", "agriculture"]
    grouped_scores = defaultdict(list)
    for ts_name in scores:
        [dataset] = [d for d in datasets if d in ts_name]
        grouped_scores[dataset].append(scores[ts_name])
    import pprint
    pprint.pprint({
        d: (round(sum(v) / len(v), 3), len(v)) for d, v in grouped_scores.items()
    })

def eval_score(answer_dir, task_dir, task_name):
    answers = {}
    ground_truths = load_ground_truth(task_dir)
    assert (all(v in {'A', 'B', 'C', 'D'} for v in ground_truths.values())
            or all(v in {'True', 'False'} for v in ground_truths.values()))
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
    return round(accuracy_rate, 3) 

#    wrong_answers = [k for k,v in answers.items() if v != ground_truths[k]]
#    if ("perturbed" not in task) and ("ts_comparison" not in task) \
#        and ("paraphrase" not in task) and ("plot_retrieval_same_domain" not in task):
#        return
#    if "with_image" in answer_dir and ("plot_retrieval_same_domain" not in task):
#        return
#    with open(os.path.join("qwen3b_wrong_answers", task + ".json"), "w") as fh:
#        json.dump(wrong_answers, fh)
 
#    score_breakdown(answers, ground_truths)

if __name__ == "__main__":
    model = sys.argv[1]
    accs = {}
    for task_dir in sorted(os.listdir(ANSWER_DIR.format(model=model))):
        task = task_dir.replace("_with_image", "").replace("_no_image", "")
        if "ts_comparison" in task:
            answer_dir_abspath = os.path.join(ANSWER_DIR.format(model=model), task_dir)
            task_dir_abspath = os.path.join(TASK_DIR, "ts_comparison", task.replace("ts_comparison_", "")) 
        else:
            answer_dir_abspath = os.path.join(ANSWER_DIR.format(model=model), task_dir)
            task_dir_abspath = os.path.join(TASK_DIR, task)
        task_acc = eval_score(answer_dir_abspath,
                   task_dir_abspath,
                   task) 
        accs[task_dir] = task_acc
    print(accs)
    with open("qa_results.json", "r") as fh:
        results = json.load(fh)
    results[model] = accs
    # Get results from closed-source models
    root_dir = Path(TASK_DIR)
    for model in ["claude-3-haiku", "Google Gemini-2.0-Flash"]:
        curr_accs = {}
        filepaths = list(root_dir.rglob(f"{model}.json"))
        for fp in filepaths:
            task_name = os.path.dirname(fp).split("/")[-1]
            if "ts_comparison" in str(fp):
                task_name = "ts_comparison_" + task_name
            with open(fp) as fh:
                z = json.load(fh)
                if isinstance(z, dict):
                    accuracy  = z["overall"]["accuracy"]
                else:
                    accuracy = z[-1]["overall accuracy"]
                curr_accs[task_name] = accuracy 

        results[model] = curr_accs
    with open("qa_results.json", "w") as fh:
        json.dump(results, fh)
     
