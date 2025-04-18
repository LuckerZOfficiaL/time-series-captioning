import json
import os
import json
import random
from factcheck_helpers import(
    load_config,
    check_whole_caption,
    check_whole_caption_confidence,
    save_file
)


def main():
    config = load_config()
    random.seed(config['general']['random_seed'])
    extraction_model = config['model']['extraction_model']
    checking_model = config['model']['checking_model']
    use_confidence = config['refinement']['use_confidence_checking']
    

    facts_path = '/home/ubuntu/thesis/data/factcheck/factcheck-GPT-benchmark.jsonl'
    #facts_path = '/home/ubuntu/thesis/data/factcheck/Factbench.jsonl'

    with open(facts_path, 'r') as file:
        facts = [json.loads(line) for line in file]


    fake_facts = []
    true_facts = []

    gt_revised_fake_facts = [] # the ground truth revised fake facts from the factcheck
    gt_revised_true_facts = [] # the ground truth revised true facts from the factcheck

    print("\nReading FactCheck Data...")
    for i in range(len(facts)):
        #print("\n\nPrompt: ", facts[0]['prompt'])
        #print("\n\nResponse: ", facts[i]['response'])
        #print("\n\nRevised Response: ", facts[i]['revised_response'])
        #print("\n\nResponse Factuality: ", facts[i]['response_factuality'])
        #print("\n\nSentences: ", facts[0]['sentences'])
        
        if facts[i]['response_factuality'] == False: # if it's a fake fact
            fake_facts.append(facts[i]['response'])
            gt_revised_fake_facts.append(facts[i]['revised_response'])
        elif facts[i]['response_factuality'] == True:
            true_facts.append(facts[i]['response'])
            gt_revised_true_facts.append(facts[i]['revised_response'])
        else: # the factuality is non defined
            pass
        #if i+1 == 35 : break


    ################################# TRACKING RESULTS FOR FAKE FACTS ##############################################
    print("\nChecking Fake Facts...")
    fake_detection_res_path = f'/home/ubuntu/thesis/source/factcheck/{extraction_model}_{checking_model}_{"conf" if use_confidence else ""}{str(config[refinement]['confidence']) if use_confidence else ""}_fake_detection_res.json'
    if not os.path.exists(fake_detection_res_path):
        fake_detection_res= {
            "correct_fake": 0,
            "i": 0
        }
        save_file(fake_detection_res, fake_detection_res_path)
    else:
        with open(fake_detection_res_path, 'r') as file:
            fake_detection_res = json.load(file)


    start_idx = fake_detection_res['i']

    for i in range(start_idx, len(fake_facts)):
        print(f"\nChecking fake fact {i+1}/{len(fake_facts)} \n{fake_facts[i]}")
        if config['refinement']['use_confidence_checking']:
            outcome, fact = check_whole_caption_confidence(fake_facts[i], extraction_model=extraction_model, checking_model=checking_model, confidence_thresh=config['refinement']['confidence_thresh'])
        else:
            outcome, fact = check_whole_caption(fake_facts[i], extraction_model=extraction_model, checking_model=checking_model, tolerate_inconclusive=True)

        if outcome == False:
            fake_detection_res['correct_fake'] += 1
            print(f"\nCorrectly detected as fake :) \nDue to: {fact}")
        else:
            print(f"\nFailed to detect as fake :(")

        fake_detection_res['i'] = i+1

        output_file = f'/home/ubuntu/thesis/source/factcheck/{extraction_model}_{checking_model}_{"conf" if use_confidence else ""}{str(config[refinement]['confidence']) if use_confidence else ""}_fake_detection_res.json'
        with open(output_file, 'w') as file:
            json.dump(fake_detection_res, file)

    ################################# TRACKING RESULTS FOR TRUE FACTS ##############################################
    print("\nChecking True Facts...")
    true_detection_res_path = f'/home/ubuntu/thesis/source/factcheck/{extraction_model}_{checking_model}_{"conf" if use_confidence else ""}{str(config[refinement]['confidence']) if use_confidence else ""}_true_detection_res.json'
    if not os.path.exists(true_detection_res_path):
        true_detection_res = {
            "correct_true": 0,
            "i": 0
        }
        save_file(true_detection_res, true_detection_res_path)
    else:
        with open(true_detection_res_path, 'r') as file:
            true_detection_res = json.load(file)

    start_idx = true_detection_res['i']
    
    for i in range(start_idx, len(true_facts)):
        print(f"\nChecking true fact {i+1}/{len(true_facts)} \n{true_facts[i]}")
        if config['refinement']['use_confidence_checking']:
            outcome, fact = check_whole_caption_confidence(true_facts[i], extraction_model=extraction_model, checking_model=checking_model, confidence_thresh=config['refinement']['confidence_thresh'])
        else:
            outcome, fact = check_whole_caption(true_facts[i], extraction_model=extraction_model, checking_model=checking_model, tolerate_inconclusive=True)

        if outcome == True:
            true_detection_res['correct_true'] += 1
            print(f"\nCorrectly recognize as true :)")
        else:
            print(f"\nFailed to recognize as true :( \nDue to: {fact}")

        true_detection_res['i'] = i+1

        output_file = f'/home/ubuntu/thesis/source/factcheck/{extraction_model}_{checking_model}_{"conf" if use_confidence else ""}{str(config[refinement]['confidence']) if use_confidence else ""}_true_detection_res.json'
        with open(output_file, 'w') as file:
            json.dump(true_detection_res, file)



    
if __name__ == "__main__":
    main()
        