import io
import contextlib
import json
from pathlib import Path
import os

from source.multi_gpu_utils import run_multi_gpu

TASK_TO_IMAGE = {
    "caption_retrieval_perturbed": [False, True],
    "paraphrase_consistency": [False],
    "plot_retrieval_same_domain": [True],
    "ts_comparison_amplitude": [False],
    "ts_comparison_bottom_earlier": [False],
    "ts_comparison_mean": [False],
    "ts_comparison_peak_earlier": [False],
    "ts_comparison_same_phenomenon": [False],
    "ts_comparison_volatility": [False],
    "ts_retrieval_perturbed": [False],
    "test": [True]
}

def run_all_tasks(eval_model_fn, data_dir, out_dir):
    filepath = os.path.join(data_dir, "tasks.json") 
    # Get things running
    run_multi_gpu(eval_model_fn, data_dir, out_dir, use_image=False)
    return

    for task_file in filepaths:
        task_dir = os.path.dirname(str(task_file))
        task_name = task_dir.split('/')[-1]
        if "ts_comparison" in task_dir:
            task_name = "ts_comparison_" + task_name

        for use_image in TASK_TO_IMAGE[task_name]:
            out_dir_name = task_name + ("_no_image" if not use_image else "_with_image")
            run_multi_gpu(eval_model_fn, task_dir, os.path.join(out_dir, out_dir_name), use_image=use_image)

PAL_PROMPT = """
### Task
{task_description}

### Instructions for the assistant
1. You are an expert coding assistant; think through the task **step-by-step**.
2. Write **Python 3.12** code ⟨inside one ```python``` block⟩ that computes the final answer.
   * Use only the Python Standard Library (e.g. you may use the `math`, `statistics` libraries).
   * Wrap everything in a `solve()` function that will be invoked to produce the final caption.
   * The code **must produce the caption string itself**. Any numerical values can be computed
     in Python and formatted into the caption string. Make sure to use any values you compute
     in the resulting caption string.  
3. The `solve()` function you write will be invoked to produce the final caption.


### Output format (exactly; no extra text, explanations, or formatting)
```python
# code that defines solve() and any desired strings
solve()
```
"""

def run_PAL_captions(eval_model_fn, prompts_file, plots_dir, out_dir):
    with open(prompts_file) as fh:
        starting_prompts = json.load(fh)
    done_ts = {x.replace(".txt", "") for x in os.listdir(out_dir)}
    print(done_ts)
    run_multi_gpu(eval_model_fn, "caption_prompts/test", out_dir, use_image=False, handler_fn=eval_PAL)

def eval_PAL(model_eval, ts_batch, device, data_dir, out_dir, use_image):
    prompts = [PAL_PROMPT.format(task_description=p["prompt_no_image"]) for p in ts_batch]
    results = model_eval(prompts=prompts, image_files=None,  device='cuda', use_image=False)
        
    for result, p in zip(results, ts_batch):
        result = result.replace('```', '').replace('python', '')
        result += "print(solve())"
        print(result)
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                exec(result)
            output = buf.getvalue()
        except Exception as e:
            print("Evaluation failed!")
            print(e)
            output = ""
        with open(os.path.join(out_dir, p["ts_name"] + ".txt"), "w") as fh:
            fh.write(output) 

