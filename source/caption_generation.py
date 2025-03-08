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
    add_facts_to_caption
)

FILE_MAPPING = {
        "air quality": "aq.json",
        "border crossing": "border_crossing.json",
        "crime": "crime.json",
        "demography": "demographics.json",
        "heart rate": "hr_data.json"
    }

SAVE_TOP_K = 5 # save the top k best captions based on the ranking
REQUEST_AUGMENTATIONS = 0 # how many times to rephrase the original prompt request?
SAMPLES = 1 # how many window samples to extract? i.e. how many time series to sample?
MODELS = ["GPT-4o-Aug", "Claude-3.5-Haiku", "Gemini-1.5-Flash", "Gemini-1.5-Pro", "DeepSeek-R1-FW"] # models used for generating captions
JUDGE_MODEL = "GPT-4o-Aug" # the model used to rank the captions
REFINEMENT_MODEL = "Gemini-1.5-Flash-Search"
REFINE_CAPTIONS = True # whether to refine the generated captions with REFINEMENT_MDOEL

def main(dataset_name):
    filepath = f"/home/ubuntu/thesis/data/processed/{FILE_MAPPING[dataset_name]}"
    with open(filepath) as f:
        json_data = json.load(f)
    
    idx = 0

    samples = get_samples(dataset_name, json_data=json_data, n=SAMPLES)

    for metadata, ts in samples:
        #print("\nMetadata: ", metadata)
        #print("\nSeries: ", ts)
        request = get_request(dataset_name, metadata, ts)
        #print("\nOriginal request: ", request)
        if REQUEST_AUGMENTATIONS > 0:
            requests = augment_request(request, n=REQUEST_AUGMENTATIONS)
        else: # do not augment the fixed-template prompt
            requests = []
        requests.append(request) # add the original fixed-template prompt too
        #print("\Requests: ")
        #for req in requests:
        #    print("\n", req)

        responses = []
        for i in range(len(requests)):
            for model in MODELS:
                response = get_response(requests[i], model=model,
                                        temperature = 0.75,
                                        top_p = 0.85
                                )
                if "DeepSeek" in model: # if it's DeepSeek, discard the reasoning text
                    match = re.search(r"Thinking\.\.\..*?>\s*\n\n(.*)", response, re.DOTALL)
                    response = match.group(1).strip() if match else None
                responses.append(response)
            #print(f"Done for request variant {i+1}.")

        if REFINE_CAPTIONS:
            print("\nCaptions are REFINED!")
            refined_captions = []
            for response in responses:
                refined_captions.append(add_facts_to_caption(response, REFINEMENT_MODEL))

        ranks = rank_responses(responses, model=JUDGE_MODEL)
        ranks = [x-1 for x in ranks]
        print("\n\nRanking: \n")
        for r in ranks:
            print(MODELS[r])

        
        #print("Ranking done: ", rank)
        print("\n\nReranked captions: \n")
        for r in ranks:
            print(MODELS[r], ":", responses[r], "\n")

        for k in range(SAVE_TOP_K):
            caption_filepath = f"/home/ubuntu/thesis/data/samples/captions/{dataset_name}_{idx}.txt" 
            save_file(responses[ranks[k]], caption_filepath)

            metadata_filepath = f"/home/ubuntu/thesis/data/samples/metadata/{dataset_name}_{idx}.json" 
            save_file(metadata, metadata_filepath)   

            series_filepath = f"/home/ubuntu/thesis/data/samples/time series/{dataset_name}_{idx}.json" 
            save_file(ts, series_filepath)   

            idx += 1

    

if __name__ == "__main__":
    main("crime")