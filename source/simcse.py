import os
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from tqdm import tqdm
import re
from collections import defaultdict
import json


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""model_path = "/home/ubuntu/thesis/data/simcse_ts_new"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path).eval().to(device)"""


model_name = "princeton-nlp/sup-simcse-roberta-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).eval().cuda()

def get_domain(fname):
    return fname.split("_")[0]




#generated_folders = [d for d in os.listdir("/home/ubuntu/thesis/data/samples/new samples no overlap/generated captions") if os.path.isdir(os.path.join("/home/ubuntu/thesis/data/samples/new samples no overlap/generated captions", d))]
#generated_folders = ["InternVL2-2B","InternVL2-2B_text", "gemini-2.0-flash_text", "gemini-2.0-flash"]
generated_folders = ["gemini-2.0-flash_text"]

for name in generated_folders:
    domain_similarities = defaultdict(list)
    
#try:
    #name = "positive pair example"
    #gt_dir = f"/home/ubuntu/thesis/data/samples/simcse experiment/simcse positive experiment gt"
    #gen_dir = f"/home/ubuntu/thesis/data/samples/simcse experiment/simcse positive experiment gen"
    
    gt_dir = f"/home/ubuntu/thesis/data/samples/new samples no overlap/test/gt_captions"
    gen_dir = f"/home/ubuntu/thesis/data/samples/new samples no overlap/generated captions/{name}"
    #gt_dir = f"/home/ubuntu/thesis/data/samples/len 300/captions"
    #gen_dir = f"/home/ubuntu/thesis/data/samples/len 300/generated captions/{name}"

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
    print(f"\n{name}\nAverage SimCSE by domain:")
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
    print(f"global average: {global_avg:.4f}")

    #with open(f"/home/ubuntu/thesis/data/samples/len 300/evaluation results/simcse_results/{name}.json", "w") as f:
    with open(f"/home/ubuntu/thesis/data/evaluation results/simcse_results/{name}.json", "w") as f:
        json.dump(domain_avg_json, f, indent=2)
        
#except Exception as e:
#    print(e)