from helpers import get_response
import os

def make_prompt(gt_caption):
    prompt = f"""You are a helpful assistant. Your task is to rephrase the following paragraph that describes a time series. You MUST strictly follow these rules:

Preserve all factual information: All numerical values, statistics (min, max, mean, etc.), trends ('increased', 'peaked'), comparisons ('higher than'), and dates must remain exactly the same.

Change the style completely: Use different sentence structures, synonyms, and grammatical constructions. Alter the tone (e.g., make it more formal or more conversational). Do not use the same phrasing as the original.

Output only the rephrased paragraph, with no additional explanation.

Here is the paragraph to rephrase: {gt_caption}"""
    return prompt

def main():
    folder_path = "/home/ubuntu/thesis/data/samples/new_samples_no_overlap/test/gt_captions"
    save_path = "/home/ubuntu/thesis/data/samples/new_samples_no_overlap/test/gt_captions_GPT-4o_paraphrased"
    os.makedirs(save_path, exist_ok=True)

    files = os.listdir(folder_path)
    keep_domains = ["air quality", "border crossing", "co2", "covid", "diet", "online retail", "road injuries"] #["agriculture", "crime", "demography", "walmart"]
    files = [filename for filename in files if filename.split("_")[0] in keep_domains]
    
    done_files = set(os.listdir(save_path))
    files = [filename for filename in files if filename not in done_files]
    
    print(f"{len(files)} to paraphrase, for domains {keep_domains}.")
    
    for i, filename in enumerate(files):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
    
            with open(file_path, 'r') as file:
                gt_caption = file.read()
                
            para_caption = get_response(make_prompt(gt_caption), model="OpenAI GPT-4o")
            
            save_file_path = os.path.join(save_path, filename)
            with open(save_file_path, 'w') as save_file:
                save_file.write(para_caption)
            
            if i%50 == 0 or i+1 == len(files):
                print(f"Done {i+1}/{len(files)}")
                    
                    

if __name__ == "__main__":
    main()