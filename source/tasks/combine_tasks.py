from pathlib import Path
import json
import os

DATA_DIR = "/shared/tsqa/CaTSBench/tasks"

def get_question_type(x: str) -> str:
    qtype = os.path.dirname(x).split('/')[-1]
    if 'ts_comparison' in x:
        qtype = 'ts_comparison_' + qtype
    return qtype

if __name__ == "__main__":
    root_dir = Path(DATA_DIR)
    filepaths = list(root_dir.rglob("tasks.json"))
    total = []
    for fp in filepaths:
        with open(fp) as fh:
            curr = json.load(fh)
        question_type = get_question_type(str(fp))
        for item in curr:
            item["task_type"] = question_type
            item["task_id"] = "_".join([question_type, item["ts_name"]])
            if "image_paths" in item:
                item["image_paths"] = [os.path.basename(x) for x in item["image_paths"]] 
        total.extend(curr)
    with open("all_tasks.json", "w") as fh:
        json.dump(total, fh)


