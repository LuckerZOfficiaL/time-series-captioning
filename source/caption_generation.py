import requests
import json
from dataset_helpers import (
    get_response, 
    get_sample, 
    get_samples, 
    get_request, 
    augment_request, 
    rank_responses,
    save_file
)

FILE_MAPPING = {
        "air quality": "aq.json",
        "border crossing": "border_crossing.json",
        "crime": "crime.json",
        "demography": "demographics.json",
        "heart rate": "hr_data.json"
    }
SAVE_TOP_K = 2 # save the top k best captions based on the ranking
AUGMENTATIONS = 2 # how many times to rephrase the original prompt request?
SAMPLES = 2 # how many window samples to extract? i.e. how many tiime series to sample?

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
        augmented_requests = augment_request(request, n=AUGMENTATIONS)
        #print("\nAugmented requests: ")
        #for req in augmented_requests:
        #    print("\n", req)

        responses = []
        for i in range(len(augmented_requests)):
            responses.append(get_response(augmented_requests[i], model="GPT-4o-Aug",
                                    temperature = 0.75,
                                    top_p = 0.85
                            )
                        )
            #print(f"Done for request variant {i+1}.")

        rank = rank_responses(responses)
        rank = [x-1 for x in rank]
        #print("Ranking done: ", rank)

        #print("\nReranked captions: ")
        #for r in rank:
        #    print("\n", responses[r])
        for k in range(SAVE_TOP_K):
            caption_filepath = f"/home/ubuntu/thesis/data/samples/captions/{dataset_name}_{idx}.txt" 
            save_file(responses[rank[k]], caption_filepath)

            metadata_filepath = f"/home/ubuntu/thesis/data/samples/metadata/{dataset_name}_{idx}.json" 
            save_file(metadata, metadata_filepath)   

            series_filepath = f"/home/ubuntu/thesis/data/samples/time series/{dataset_name}_{idx}.json" 
            save_file(ts, series_filepath)   

            idx += 1

    

if __name__ == "__main__":
    main("demography")