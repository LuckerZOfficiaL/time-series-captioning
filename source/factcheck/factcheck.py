import json
from factcheck_helpers import(
    are_semantically_equivalent,
    are_semantically_conflicting,
    is_semantically_contained,
    refine_caption_with_corrected_facts,
    load_config
)



def main():
    config = load_config()

    facts_path = '/home/ubuntu/thesis/data/factcheck/factcheck-GPT-benchmark.jsonl'#'/home/ubuntu/thesis/data/factcheck/Factbench.jsonl'

    with open(facts_path, 'r') as file:
        facts = [json.loads(line) for line in file]
    

    fake_facts = []
    true_facts = []

    gt_revised_fake_facts = [] # the ground truth revised fake facts from the factcheck
    gt_revised_true_facts = [] # the ground truth revised true facts from the factcheck

    for i in range(len(facts)):
        #print("\n\nPrompt: ", facts[0]['prompt'])
        #print("\n\nResponse: ", facts[i]['response'])
        #print("\n\nRevised Response: ", facts[i]['revised_response'])
        #print("\n\nResponse Factuality: ", facts[i]['response_factuality'])
        #print("\n\nSentences: ", facts[0]['sentences'])
        
        if facts[i]['response_factuality'] == False: # if it's a fake fact
            fake_facts.append(facts[i]['response'])
            gt_revised_fake_facts.append(facts[i]['revised_response'])
        else:
            true_facts.append(facts[i]['response'])
            gt_revised_true_facts.append(facts[i]['revised_response'])

        if i+1 == 5 : break
    
    llm_revised_fake_facts = [] # contains revised fake facts, revised by our method
    llm_revised_true_facts = [] # contains revised true facts, revised by our method


    ############### Using Our Method #####################
    for i in range(len(fake_facts)):
        revised_fact = refine_caption_with_corrected_facts(fake_facts[i],
                                                            model=config['model']['refinement_model'],
                                                            correction_method="llm",
                                                            return_corrected_facts=False,
                                                            skip_numeric=False,
                                                            extract_sentences=True)
        llm_revised_fake_facts.append(revised_fact)
        print(f"Refined fake fact {i+1}/{len(fake_facts)}")

    for i in range(len(true_facts)):
        revised_fact = refine_caption_with_corrected_facts(true_facts[i],
                                                            model=config['model']['refinement_model'],
                                                            correction_method="llm",
                                                            return_corrected_facts=False,
                                                            skip_numeric=False,
                                                            extract_sentences=True)
        llm_revised_true_facts.append(revised_fact)
        print(f"Refined true fact {i+1}/{len(true_facts)}")
    


    ############### Metric EVALUATION #####################
    inclusions = 0
    conflicts = 0
    equivalences = 0

    print("\nEvaluating Fake Facts...")
    for llm_revised_fake_fact, gt_revised_fake_facts in zip(llm_revised_fake_facts, gt_revised_fake_facts):
        if are_semantically_equivalent(llm_revised_fake_fact, gt_revised_fake_facts):
            equivalences += 1
        elif is_semantically_contained(gt_revised_fake_facts, llm_revised_fake_fact):
            inclusions += 1
        elif are_semantically_conflicting(llm_revised_fake_fact, gt_revised_fake_facts):
            conflicts += 1
            print(f"\nCONFLICT! \nLLM: {llm_revised_fake_fact} \nGT: {gt_revised_fake_facts}")
        else: print(f"Nothing detected! \nLLM: {llm_revised_true_fact} \nGT: {gt_revised_true_facts}")
        
    print("\nFake Facts:")
    print(f"Equivalences: {equivalences}/{len(fake_facts)}")
    print(f"Inclusions: {inclusions}/{len(fake_facts)}")
    print(f"Conflicts: {conflicts}/{len(fake_facts)}")



    inclusions = 0
    conflicts = 0
    equivalences = 0
    print("\nEvaluating True Facts...")
    for llm_revised_true_fact, gt_revised_true_facts in zip(llm_revised_true_facts, gt_revised_true_facts):
        if are_semantically_equivalent(llm_revised_true_fact, gt_revised_true_facts):
            equivalences += 1
        elif is_semantically_contained(gt_revised_true_facts, llm_revised_true_fact):
            inclusions += 1
        elif are_semantically_conflicting(llm_revised_true_fact, gt_revised_true_facts):
            conflicts += 1
            print(f"\nCONFLICT! \nLLM: {llm_revised_true_fact} \nGT: {gt_revised_true_facts}")
        else: print(f"Nothing detected! \nLLM: {llm_revised_true_fact} \nGT: {gt_revised_true_facts}")
        

    print("\n\nTrue Facts:")
    print(f"Equivalences: {equivalences}/{len(true_facts)}")
    print(f"Inclusions: {inclusions}/{len(true_facts)}")
    print(f"Conflicts: {conflicts}/{len(true_facts)}")


if __name__ == "__main__":
    main()


