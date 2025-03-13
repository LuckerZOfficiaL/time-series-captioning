# This script requires that process_data.py is run already. This script generates samples from the given processed data. Each sample consists of (time series, metadata, caption).

import requests
import json
import re
from helpers import (
    get_response, 
    get_sample, 
    get_samples, 
    get_request, 
    augment_request, 
    rank_responses,
    save_file,
    add_facts_to_caption,
    generate_line_plot
)


FILE_MAPPING = {
        "air quality": "aq.json",
        "border crossing": "border_crossing.json",
        "crime": "crime.json",
        "demography": "demographics.json",
        "heart rate": "hr_data.json"
    }

REQUEST_AUGMENTATIONS = 0 # how many times to rephrase the original prompt request?
N_SAMPLES = 3 # how many window samples to extract per dataset? i.e. how many time series to sample?
ALL_MODELS = ["Google Gemini-2.0-Flash", "OpenAI GPT-4o", "Anthropic Claude-3.5", "GPT-4o", "Claude-3.5-Haiku", "Gemini-1.5-Flash", "Gemini-1.5-Pro", "DeepSeek-R1-FW"] # available model choices, the first two are from official APIs
MODELS = ["OpenAI GPT-4o", "Anthropic Claude-3.5", "Google Gemini-2.0-Flash"] # models to use for generating captions
JUDGE_MODEL = "OpenAI GPT-4o" # the model used to rank the captions
SAVE_TOP_K = 0 # save the top k best captions based on the ranking, if it's 0 or negative, don't do top-k. If top-k is on, caption ranking is invoked
EMBEDDING_MODEL = "all-MiniLM-L6-v2" # the embedding model used for RAG
RAG_TOP_K = 5 # how many top-relevant facts to retrieve from the bank
RAG = False # whether to apply RAG on caption generation



def main(dataset_names):
    for dataset_name in dataset_names:
        print("\nGenerating captions for", dataset_name)
        filepath = f"/home/ubuntu/thesis/data/processed/{FILE_MAPPING[dataset_name]}"
        with open(filepath) as f:
            json_data = json.load(f)
        
        idx = 0

        samples = get_samples(dataset_name, json_data=json_data, n=N_SAMPLES)

        requests = []
        for metadata, ts in samples: # this loop accumulates all requests, across different samples
            #print("\nMetadata: ", metadata)
            #print("\nSeries: ", ts)
            #print(metadata, ts)
            this_sample_request = get_request(dataset_name, metadata, ts)
            requests.append(this_sample_request)

            if REQUEST_AUGMENTATIONS > 0:
                this_sample_requests = augment_request(this_sample_request, n=REQUEST_AUGMENTATIONS)
                requests.extend(this_sample_requests)


        if RAG:
            print("\nApplying RAG to the prompts.")
            embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        
            with open("/home/ubuntu/thesis/data/fact bank/all_facts.txt", "r") as file: # Load all facts 
                all_facts_list = file.read().splitlines()

            all_facts_emb = torch.load("/home/ubuntu/thesis/data/fact bank/all_facts_emb.pth").cpu() # Load all fact embeddings 

            for i in range(len(requests)):
                prompt_embedding = model.encode(requests[i])
                requests[i] = augment_prompt_with_facts(requests[i], all_facts_list, all_facts_emb, embedding_model)
            

        responses = [] 
        for model in MODELS: # for each model, all requests are asked and the responses are collected
            model_responses = get_response(requests, model=model,
                                    temperature = 0.75,
                                    top_p = 0.85)
            responses.extend(model_responses)
        #print("\n\Responses: ", responses)
        

        if SAVE_TOP_K > 0:
            ranks = rank_responses(responses, model=JUDGE_MODEL)
            ranks = [x-1 for x in ranks] #to make the rank start from index 0 instead of 1

            for k in range(SAVE_TOP_K):
                caption_filepath = f"/home/ubuntu/thesis/data/samples/captions/{"rag" if RAG else "raw"}/{dataset_name}_{idx}.txt" 
                save_file(responses[rank[k]], caption_filepath)

                metadata_filepath = f"/home/ubuntu/thesis/data/samples/metadata/{dataset_name}/{dataset_name}_{idx}.json" 
                save_file([meta_and_ts[0] for meta_and_ts in samples][ranks[k]], metadata_filepath)   

                series_filepath = f"/home/ubuntu/thesis/data/samples/time series/{dataset_name}/{dataset_name}_{idx}.json" 
                save_file([meta_and_ts[1] for meta_and_ts in samples][ranks[k]], series_filepath) 

                idx += 1
        else: # just save all responses without ranking and without selecting top-k
            for i in range(len(responses)):
                caption_filepath = f"/home/ubuntu/thesis/data/samples/captions/{dataset_name}_{idx}.txt" 
                save_file(responses[i], caption_filepath)

                metadata_filepath = f"/home/ubuntu/thesis/data/samples/metadata/{dataset_name}_{idx}.json" 
                save_file([meta_and_ts[0] for meta_and_ts in samples][i%len(samples)], metadata_filepath)   

                series_filepath = f"/home/ubuntu/thesis/data/samples/time series/{dataset_name}_{idx}.json" 
                save_file([meta_and_ts[1] for meta_and_ts in samples][i%len(samples)], series_filepath) 

                idx += 1


if __name__ == "__main__":
    dataset_names = list(FILE_MAPPING.keys())
    main(dataset_names)

    # Test Code
    """prompts = [
        "What is 1+1 equal to?",
        "What is 2+2 equal to?",
        "What is the color of an orange?"
    ]

    print(get_response(prompt=prompts,
                        model = "GPT-4o",
                        temperature = 0.75,
                        top_p = 0.85,
                    ))"""