import json

import numpy as np
import pandas as pd

NAME_REMAP = {
    "internvl25": "InternVL-2.5",
    "phi": "Phi-4",
    "qwen": "Qwen2.5-Omni",
    "claude-3-haiku": "Claude-3-Haiku",
    "Google Gemini-2.0-Flash": "Gemini-2.0-Flash",
    "llava": "LLaVa v1.6 Mistral"
}
def bold_row_max(row, tol=1e-12):
    """Return a list of strings where the maximum in *row*
       is wrapped with \textbf{...}.  Ties are all bolded."""
    m = row.max()
    return [
        rf"\textbf{{{v:.2f}}}" if np.isclose(v, m, atol=tol) else f"{v:.2f}"
        for v in row
    ]

if __name__ == "__main__":
    with open("qa_results.json") as fh:
        scores = json.load(fh)
    scores_rename = {}
    for k in scores:
        scores_rename[NAME_REMAP.get(k, k)] = scores[k]
    scores = scores_rename
    for k in scores:
        scores[k].pop("caption_retrieval_perturbed_with_image", None)
        scores[k].pop("ts_comparison_bottom_earlier_no_image", None)
        scores[k].pop("ts_comparison_bottom_earlier", None)
        scores[k].pop("ts_comparison_same_phenomenon_no_image", None)
        scores[k].pop("ts_comparison_same_phenomenon", None)

        new_scores = {}
        for t in scores[k]:
            new_t = t.replace("retrieval", "matching").replace("_no_image", "").replace("_with_image", "").replace("_perturbed", "")
            new_t = new_t.replace("_same_domain", "").replace("_", " ")
            new_t = new_t.title().replace("Ts", "TS")
            new_scores[new_t] = scores[k][t]
        scores[k] = new_scores
    import pprint
    pprint.pprint(scores)
    df = pd.DataFrame(scores).T
    
    # Optional: sort rows (tasks) alphabetically for readability
    #df = df.sort_index()

    # apply along rows, keep original column names
    pretty = df.apply(bold_row_max, axis=1, result_type="expand")
    pretty.columns = df.columns       # restore column headers
    
    latex_table = pretty.to_latex(
        escape=False,                 # <- must be False so \textbf passes through
        index=True,                   # keep row labels
        caption="Model accuracy on each time-series evaluation task",
        label="tab:accuracy_by_model_task",
        column_format="l" + "c"*len(pretty.columns)  # left row label + centered cols
    )
    
    print(latex_table) 
