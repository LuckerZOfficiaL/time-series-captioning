import csv
import json
import os
from pathlib import Path
import random
import re

TASK_DIR = "data/samples/new samples no overlap/tasks"
OUT_DIR = "data/samples/new samples no overlap/hard_questions"
DOWNSAMPLE_OUT_DIR = "data/samples/new samples no overlap/hard_questions_small"
WRONG_ANSWER_DIR = "qwen3b_wrong_answers"

def make_hard_questions(wrong_answer_dir):
    for fh in os.listdir(wrong_answer_dir):
        with open(os.path.join(wrong_answer_dir, fh)) as fh2:
            ts_names = json.load(fh2)
        if "ts_comparison" in fh:
            task_name = fh.replace("ts_comparison_", "").replace(".json", "")
            orig_file = os.path.join(TASK_DIR, 'ts_comparison', task_name, "tasks.json")
            out_dir = os.path.join(OUT_DIR, 'ts_comparison', task_name)
        else:
            task_name = fh.replace(".json", "")
            orig_file = os.path.join(TASK_DIR, task_name, "tasks.json")
            out_dir = os.path.join(OUT_DIR, task_name)
        with open(orig_file) as fh3:
            all_prompts = json.load(fh3)
        hard_prompts = [p for p in all_prompts if p["ts_name"] in ts_names]
        total_q += len(hard_prompts)
        print(f"Task name: {task_name}, num questions: {len(hard_prompts)}")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "tasks.json"), "w") as fh4:
            json.dump(hard_prompts, fh4)
    print(f"Total questions: {total_q}")

def downsample_hard_questions(downsample_dir, task_count=100, ts_comparison_count=40):
    random.seed(414)
    root_dir = Path(OUT_DIR)
    filepaths = list(root_dir.rglob("tasks.json"))
    for fp in filepaths:
        fp = str(fp)
        with open(fp) as fh:
            all_prompts = json.load(fh)
        dirname = os.path.dirname(fp).replace(OUT_DIR, downsample_dir)
        os.makedirs(dirname, exist_ok=True) 
        downsample_count = ts_comparison_count if "ts_comparison" in dirname else task_count
        downsampled_prompts = random.sample(all_prompts, min(downsample_count, len(all_prompts))) 
        with open(os.path.join(dirname, 'tasks.json'), 'w') as fh:
            json.dump(downsampled_prompts, fh)

PATTERN = r"(?P<label>(\([ABCD]\)|[ABCD]:))"

def extract_sections(text):
    # Matches label like (A), A:, A:, (B): etc.
    # Find all matches and their positions
    matches = list(re.finditer(PATTERN, text))
    sections = {}

    for i in range(len(matches)):
        label_raw = matches[i].group("label")
        label = re.sub(r"[^\w]", "", label_raw)  # strip () and :
        start = matches[i].end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections[label] = text[start:end].strip()

    return sections

def make_mech_turk_csv():
    csv_path = "human_eval.csv"
    fieldnames = [
        "Question ID", "Question Type", "Question",
        "Option 1", "Option 2", "Option 3", "Option 4",
        "Ground Truth"
    ]
    option_list = ["Option 1", "Option 2", "Option 3", "Option 4"]
    root_dir = Path(DOWNSAMPLE_OUT_DIR)
    filepaths = list(root_dir.rglob("tasks.json"))
    all_questions = []
    for fp in filepaths:
        fp = str(fp) 
        with open(fp) as fh:
            prompts = json.load(fh)
        task_name = fp.split('/')[-2]
        for p in prompts:
            gt = p['ground_truth']
            prompt_txt = (p['prompt'] if 'prompt' in p else p['prompt_no_image'])
            prompt_txt = prompt_txt.split('You must')[0]
            if "plot_retrieval" in task_name:
                s1 = 'Here are four plots of different time series'
                prompt_txt = prompt_txt.split(s1)[0] + ' Which plot corresponds to the time series provided above?'
                options = {f"Option {i+1}": os.path.basename(p["image_paths"][i]) for i in range(len(p["image_paths"]))}
                options["Ground Truth"] = os.path.basename(p["image_paths"][['A', 'B', 'C', 'D'].index(p["ground_truth"])])
            elif gt in {'True', 'False'}:
                options = {
                    "Option 1": "True",
                    "Option 2": "False",
                    "Option 3": "",
                    "Option 4": "",
                    "Ground Truth": gt
                }
            elif "ts_comparison" in fp:
                opts = extract_sections(prompt_txt)
                first_index = re.search(PATTERN, prompt_txt).start()
                prompt_txt = prompt_txt[:first_index]
                options = {
                    "Option 1": opts['A'],
                    "Option 2": opts['B'],
                    "Option 3": "", 
                    "Option 4": "",
                    "Ground Truth": gt
                }
            elif gt in {'A', 'B', 'C', 'D'}:
                opts = extract_sections(prompt_txt)
                first_index = re.search(PATTERN, prompt_txt).start()
                prompt_txt = prompt_txt[:first_index]
                options = {
                    "Option 1": opts['A'],
                    "Option 2": opts['B'],
                    "Option 3": opts['C'],
                    "Option 4": opts['D'],
                    "Ground Truth": gt
                }


            curr_q = {
                "Question ID": task_name + "_" +  p['ts_name'],
                "Question Type": task_name,
                "Question": prompt_txt,
            }
            curr_q.update(options)
            all_questions.append(curr_q)
   
    with open("human_eval.json", "w") as fh:
        json.dump(all_questions, fh)
 
    with open(csv_path, mode="w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_questions)
 

total_q = 0
if __name__ == "__main__":
    #downsample_hard_questions("data/samples/new samples no overlap/hard_questions_medium", task_count=1000, ts_comparison_count=1000)
    make_mech_turk_csv()
