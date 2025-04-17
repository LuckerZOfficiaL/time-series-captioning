# This script requires that process_data.py is run already. This script generates samples from the given processed data. Each sample consists of (time series, metadata, caption).

import requests
import json
import re
import random
import torch
from sentence_transformers import SentenceTransformer
import os
from helpers import (
    get_response, 
    get_sample, 
    get_samples, 
    get_request, 
    augment_request, 
    rank_responses,
    save_file,
    add_facts_to_caption,
    generate_line_plot,
    embed_sentences,
    augment_prompt_with_rag,
    extract_years,
    get_relevant_facts,
    load_config
)


FILE_MAPPING = {
        "air quality": "aq.json",
        "border crossing": "border_crossing.json",
        "crime": "crime.json",
        "demography": "demographics.json",
        "road injuries": "road_injuries.json",
        "covid": "covid.json",
        "co2": "co2.json",
        "diet": "diet.json",
        "walmart": "walmart.json",
        "online retail": "online_retail.json",
        "agriculture": "agricultural_productivity.json",
        
        
        #"heart rate": "hr_data.json"
    }

"""REQUEST_AUGMENTATIONS = 0 # how many times to rephrase the original prompt request?
N_SAMPLES = 3 # how many window samples to extract per dataset? i.e. how many time series to sample?
ALL_MODELS = ["Google Gemini-2.0-Flash", "OpenAI GPT-4o", "Anthropic Claude-3.5", "GPT-4o", "Claude-3.5-Haiku", "Gemini-1.5-Flash", "Gemini-1.5-Pro", "DeepSeek-R1-FW"] # available model choices, the first two are from official APIs
MODELS = ["OpenAI GPT-4o", "Anthropic Claude-3.5", "Google Gemini-2.0-Flash"] # models to use for generating captions
RANKING_MODEL = "OpenAI GPT-4o" # the model used to rank the captions
SAVE_TOP_K = 0 # save the top k best captions based on the ranking, if it's 0 or negative, don't do top-k. If top-k is on, caption ranking is invoked
EMBEDDING_MODEL = "all-MiniLM-L6-v2" # the embedding model used for RAG
RAG_TOP_K = 5 # how many top-relevant facts to retrieve from the bank
RAG = True # whether to apply RAG on caption generation, it will only retrieve the facts that are temporally relevant to the prompt request.
BIN_PERIOD = 10 # the size of the bins. Each bin represents one period of time"""



def main():
    config = load_config()
    random.seed(config['general']['random_seed'])
    is_train = config['general']['is_train']
    
    if is_train: 
        with open("/home/ubuntu/thesis/data/samples/train_data_sizes.json", 'r') as f:
            data_sizes = json.load(f)
    else:
         with open("/home/ubuntu/thesis/data/samples/test_data_sizes.json", 'r') as f:
            data_sizes = json.load(f)
            
    dataset_names = config['data']['dataset_names']
    request_augmentations = config['data']['request_augmentations']
    #n_samples = config['data']['n_samples']
    #all_models = config['model']['all_models']
    used_models = config['model']['used_models']
    ranking_model = config['model']['ranking_model']
    save_top_k = config['data']['save_top_k']
    embedding_model = config['model']['embedding_model']
    use_rag = config['rag']['use_rag']
    rag_top_k = config['rag']['rag_top_k']
    bin_years = config['bank']['bin_years']



    if use_rag: embedding_model = SentenceTransformer(embedding_model)

    for dataset_name in dataset_names:
        n_samples = data_sizes[dataset_name]
        
        print(f"\nGenerating{" RAG " if use_rag else " "} {n_samples} samples for", dataset_name)
        filepath = f"/home/ubuntu/thesis/data/processed/{FILE_MAPPING[dataset_name]}"
        with open(filepath) as f:
            json_data = json.load(f)
        
        idx = 0

        samples = get_samples(dataset_name, json_data=json_data, n=n_samples, is_train=is_train)

        print(f"{dataset_name} has {len(samples)} samples.\nNow generating {len(used_models)*len(samples)} {"train" if is_train else "test"} captions out of them using {len(used_models)} oracles...")
        requests = []
        for i in range(len(samples)): 
            metadata, ts = samples[i]
            #print(f"Generated {"RAG" if use_rag else ""} prompt request for {i+1}/{len(samples)}")
            #print("\nMetadata: ", metadata)
            #print("\nSeries: ", ts)
            #print(metadata, ts)
            this_sample_request = get_request(dataset_name, metadata, ts, external_knowledge=config['data']['external_knowledge'])
            
            if use_rag:
                start_keys = [key for key in metadata if "start" in key] # this is because different datasets have different keys that denominate the start time entry
                end_keys = [key for key in metadata if "end" in key] # same for end time

                start_key = start_keys[0] if len(start_keys) == 1 else [key for key in start_keys if "year" in key][0]
                end_key = end_keys[0] if end_keys != [] else None
                
                start_time = metadata.get(start_key) # get the start time of the series
                end_time = metadata.get(end_key) if end_key is not None else None # get the end time of the series
         
                start_year = 0 # set to 0 as default, since 0 will retrieve all general facts with no year association
                if start_time is not None: 
                    #print("\nstart time:", start_time)
                    try:
                        start_year = extract_years(start_time)[0]
                    except IndexError as e:
                        print(e)
                        print("start_time: ", start_time)
                end_year = 0 
                if end_time is not None:
                    #print("end time:", end_time)
                    try: 
                        end_year = extract_years(end_time)[0]
                    except IndexError as e:
                        print(e)
                        print("end_time: ", end_time)
                 
                #print("\nStart keys",start_keys)
                #print("End keys",end_keys)
                #print("Start year",start_year)
                #print("End year", end_year)

                relevant_facts_list = get_relevant_facts(start_year, end_year, bin_years)
                relevant_facts_list.extend(get_relevant_facts(0, 0, bin_years)) # beyond the time-relevant facts, also add the general facts that ain't associated with a year. Recall that 0 is the key of general facts.
                relevant_facts_list = list(set(relevant_facts_list)) # remove duplicates if there are any
                relevant_facts_emb = embed_sentences(relevant_facts_list, model=embedding_model)
                this_sample_request = augment_prompt_with_rag(this_sample_request,
                                                                relevant_facts_list,
                                                                relevant_facts_emb,
                                                                embedding_model=embedding_model,
                                                                retrieve_k=rag_top_k)           
            requests.append(this_sample_request)

            if request_augmentations > 0:
                this_sample_requests = augment_request(this_sample_request, n=request_augmentations)
                requests.extend(this_sample_requests)

        """if RAG: # augment the requests with some K retrieved facts
            embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        
            with open("/home/ubuntu/thesis/data/fact bank/all_facts.txt", "r") as file: # Load all facts 
                all_facts_list = file.read().splitlines()

            all_facts_emb = torch.load("/home/ubuntu/thesis/data/fact bank/all_facts_emb.pth").cpu() # Load all fact embeddings 

            for i in range(len(requests)):
                requests[i] = augment_prompt_with_rag(requests[i], 
                                                        all_facts_list, 
                                                        all_facts_emb, 
                                                        embedding_model, 
                                                        retrieve_k = RAG_TOP_K)"""
            

        responses = [] 
        for model in used_models: # for each model, all requests are asked and the responses are collected
            model_responses = get_response(requests, model=model,
                                    temperature = 0.3,
                                    top_p = 0.85)
            responses.extend(model_responses)
        #print("\n\Responses: ", responses)
        
        
        if use_rag:
            save_folder = "/home/ubuntu/thesis/data/samples/captions/rag"
        elif config['data']['external_knowledge'] == False:
            save_folder = "/home/ubuntu/thesis/data/samples/new samples with overlap/all/gt_captions"
        else: save_folder = "/home/ubuntu/thesis/data/samples/captions/raw"


        if save_top_k > 0:
            ranks = rank_responses(responses, model=ranking_model)
            ranks = [x-1 for x in ranks] #to make the rank start from index 0 instead of 1

            for k in range(save_top_k):
                caption_filepath = f"{save_folder}/{dataset_name}_{idx}.txt" 
                save_file(responses[rank[k]], caption_filepath)

                metadata_filepath = f"/home/ubuntu/thesis/data/samples/metadata/{dataset_name}/{dataset_name}_{idx}.json" 
                save_file([meta_and_ts[0] for meta_and_ts in samples][ranks[k]], metadata_filepath)   

                series_filepath = f"/home/ubuntu/thesis/data/samples/time series/{dataset_name}/{dataset_name}_{idx}.txt" 
                save_file([meta_and_ts[1] for meta_and_ts in samples][ranks[k]], series_filepath) 

                idx += 1
        else: # just save all responses without ranking and without selecting top-k
            for i in range(len(responses)):
                caption_filepath = f"{save_folder}/{dataset_name}_{idx}_{"train" if is_train else "test"}.txt" 
                save_file(responses[i], caption_filepath)

                metadata_filepath = f"/home/ubuntu/thesis/data/samples/new samples with overlap/all/metadata/{dataset_name}_{idx}_{"train" if is_train else "test"}.json" 
                save_file([meta_and_ts[0] for meta_and_ts in samples][i%len(samples)], metadata_filepath)   

                series_filepath = f"/home/ubuntu/thesis/data/samples/new samples with overlap/all/time series/{dataset_name}_{idx}_{"train" if is_train else "test"}.txt" 
                save_file([meta_and_ts[1] for meta_and_ts in samples][i%len(samples)], series_filepath) 

                idx += 1
        print(f"Done for {dataset_name}")


if __name__ == "__main__":
    main()

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