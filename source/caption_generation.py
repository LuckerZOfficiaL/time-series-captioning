import requests
import json
from dataset_helpers import get_response, get_sample, get_samples, get_request, augment_request, rank_responses

FILE_MAPPING = {
        "air quality": "aq.json",
        "border crossing": "border_crossing.json",
        "crime": "crime.json",
        "demographics": "demographics.json",
        "heart rate": "hr_data.json"
    }

def main(dataset_name):
    filepath = f"/home/ubuntu/thesis/data/processed/{FILE_MAPPING[dataset_name]}"
    with open(filepath) as f:
        json_data = json.load(f)
    
    print(get_response(prompt="Please continue the sequence: 1, 4, 9", use_openAI=False))
    """metadata, ts = get_sample(dataset_name, json_data=json_data)
    print("\nMetadata: ", metadata)
    print("\nSeries: ", ts)
    request = get_request(dataset_name, metadata, ts)
    print("\nOriginal request: ", request)
    augmented_requests = augment_request(request, n=3)
    print("\nAugmented requests: ")
    for req in augmented_requests:
        print("\n", req)

    responses = []
    for i in range(len(augmented_requests)):
        responses.append(get_response(augmented_requests[i], model="GPT-4o-Aug",
                                temperature = 0.75,
                                top_p = 0.85
                        )
                    )
        print(f"Done for request variant {i+1}.")

    rank = rank_responses(responses)
    rank = [x-1 for x in rank]
    print("Ranking done: ", rank)

    print("\nReranked captions: ")
    for r in rank:
        print("\n", responses[r])"""
    
    

if __name__ == "__main__":
    main("air quality")