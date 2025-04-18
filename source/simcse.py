import os
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from tqdm import tqdm
import re
from collections import defaultdict
import json

name = "internvl-finetune11"

gt_dir = "/home/ubuntu/thesis/data/samples/new samples no overlap/test/gt_captions"
gen_dir = "/home/ubuntu/thesis/data/samples/new samples no overlap/generated captions/internvl-finetune-pratham"

model_path = "/home/ubuntu/thesis/data/simcse_ts"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path).eval().to(device)

# def is_multiple_of_3(fname):
#     match = re.search(r"_(\d+)\.txt$", fname)
#     return match and int(match.group(1)) % 3 == 0

def get_domain(fname):
    return fname.split("_")[0]

domain_similarities = defaultdict(list)

gt_all = {f for f in os.listdir(gt_dir) if f.endswith(".txt")}
gen_all = set(os.listdir(gen_dir))
common_files = sorted(gt_all & gen_all)

for fname in tqdm(common_files):
    gt_path = os.path.join(gt_dir, fname)
    gen_path = os.path.join(gen_dir, fname)

    if not os.path.exists(gen_path):
        continue

    with open(gt_path, "r") as f:
        gt_text = f.read().strip()
    with open(gen_path, "r") as f:
        gen_text = f.read().strip()

    texts = [gt_text, gen_text]
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)

    with torch.no_grad():
        embeddings = model(**inputs).pooler_output
        embeddings = F.normalize(embeddings, p=2, dim=1)
        similarity = F.cosine_similarity(embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0)).item()

    domain = get_domain(fname)
    domain_similarities[domain].append(similarity)


domain_avg_json = {}
print("\average SimCSE by domain:")
for domain, sims in sorted(domain_similarities.items()):
    avg = sum(sims) / len(sims)
    domain_avg_json[domain] = {
        "avg_sim": round(avg, 4)
    }
    print(f"{domain:<20}: {avg:.4f}  (n={len(sims)})")

all_sims = [sim for sims in domain_similarities.values() for sim in sims]
global_avg = sum(all_sims) / len(all_sims)
domain_avg_json["global_avg"] = {
    "avg_sim": round(global_avg, 4)
}
print(f"\nglobal average: {global_avg:.4f}")

with open(f"/home/ubuntu/thesis/data/evaluation results/simcse_results/{name}.json", "w") as f:
    json.dump(domain_avg_json, f, indent=2)
