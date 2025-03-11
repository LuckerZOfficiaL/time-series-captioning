# This script requires that process_data.py is run already. This script generates samples from the given processed data. Each sample consists of (time series, metadata, caption).

import requests
import json
import re
from dataset_helpers import (
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
ALL_MODELS = ["OpenAI GPT-4o", "Anthropic Claude-3.5", "GPT-4o", "Claude-3.5-Haiku", "Gemini-1.5-Flash", "Gemini-1.5-Pro", "DeepSeek-R1-FW"] # available model choices, the first two are from official APIs
MODELS = ["OpenAI GPT-4o", "Anthropic Claude-3.5"] # models to use for generating captions
JUDGE_MODEL = "OpenAI GPT-4o" # the model used to rank the captions
REFINEMENT_MODEL = "Gemini-2.0-Flash-Search"
REFINE_CAPTIONS = False # whether to refine the generated captions with REFINEMENT_MDOEL (deprecated: keep it as False and run the script caption_refinement.py to do this separately)
SAVE_TOP_K = 0 # save the top k best captions based on the ranking, if it's 0 or negative, don't do top-k. If top-k is on, caption ranking is invoked



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

            #print("\Requests: ")
            #for req in requests:
            #    print("\n", req)
        #print("\n\nRequests: ", requests)

        responses = [] 
        for model in MODELS: # for each model, all requests are asked and the responses are collected
            model_responses = get_response(requests, model=model,
                                    temperature = 0.75,
                                    top_p = 0.85)
            responses.extend(model_responses)
        #print("\n\Responses: ", responses)
        

        if REFINE_CAPTIONS:
            print("\nCaptions are getting REFINED!")
            refined_captions = []
            for response in responses:
                refined_captions.append(add_facts_to_caption(response, REFINEMENT_MODEL))

        if SAVE_TOP_K > 0:
            ranks = rank_responses(responses, model=JUDGE_MODEL)
            ranks = [x-1 for x in ranks] #to make the rank start from index 0 instead of 1

            for k in range(SAVE_TOP_K):
                caption_filepath = f"/home/ubuntu/thesis/data/samples/captions/{dataset_name}_{idx}.txt" 
                save_file(responses[rank[k]], caption_filepath)

                metadata_filepath = f"/home/ubuntu/thesis/data/samples/metadata/{dataset_name}_{idx}.json" 
                save_file([meta_and_ts[0] for meta_and_ts in samples][ranks[k]], metadata_filepath)   

                series_filepath = f"/home/ubuntu/thesis/data/samples/time series/{dataset_name}_{idx}.json" 
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