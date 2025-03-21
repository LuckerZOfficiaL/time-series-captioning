import json
import sys
import os
from factcheck_helpers import(
    are_semantically_equivalent,
    are_semantically_conflicting,
    is_semantically_contained,
    refine_caption_with_corrected_facts,
    load_config,
    compare_correctness,
    save_file
)

class DualStdout: # class for simultaneously printing and writing in a log
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = log_file

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush() # makes sure that the log is written right away

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def main():
    config = load_config()

    if config['factcheck']['start_from_files'] == False: # if we want to re-generate the data instead of reading factsfrom the files
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
        
        llm_revised_fake_facts = [] # contains revised fake facts, revised by our method
        llm_revised_true_facts = [] # contains revised true facts, revised by our method
    

        ############### Using Our Method #####################
        print("\nFalsity Detection & Correction Running...")
        if os.path.exists('/home/ubuntu/thesis/source/factcheck/fake_count.txt'):
            with open('/home/ubuntu/thesis/source/factcheck/fake_count.txt', 'r') as file:
                start_idx = int(file.read())
        else:
            start_idx = 0

        for i in range(start_idx, len(fake_facts)):
            print(f"Refing fake fact {i+1}/{len(fake_facts)}")
            revised_fact = refine_caption_with_corrected_facts(fake_facts[i],
                                                                model=config['model']['refinement_model'],
                                                                correction_method="llm",
                                                                return_corrected_facts=False,
                                                                skip_numeric=False,
                                                                extract_sentences=True,
                                                                )
            llm_revised_fake_facts.append(revised_fact)
            
            if config['factcheck']['save_files']:
                save_file(str(i+1), '/home/ubuntu/thesis/source/factcheck/fake_count.txt', 'w')
                save_file(llm_revised_fake_facts[-1]+"\n"+"_"*80+"\n", f'/home/ubuntu/thesis/source/factcheck/{config['model']['refinement_model']}_llm_revised_fake_facts.txt', 'a')




        if os.path.exists('/home/ubuntu/thesis/source/factcheck/true_count.txt'):
            with open('/home/ubuntu/thesis/source/factcheck/true_count.txt', 'r') as file:
                start_idx = int(file.read())
        else:
            start_idx = 0
        for i in range(start_idx, len(true_facts)):
            print(f"Refining true fact {i+1}/{len(true_facts)}")
            revised_fact = refine_caption_with_corrected_facts(true_facts[i],
                                                                model=config['model']['refinement_model'],
                                                                correction_method="llm",
                                                                return_corrected_facts=False,
                                                                skip_numeric=False,
                                                                extract_sentences=True)
            llm_revised_true_facts.append(revised_fact)
            
            if config['factcheck']['save_files']:
                save_file(str(i+1), '/home/ubuntu/thesis/source/factcheck/true_count.txt', 'w')
                save_file(llm_revised_true_facts[-1]+"\n"+"_"*80+"\n", f'/home/ubuntu/thesis/source/factcheck/{config['model']['refinement_model']}_llm_revised_true_facts.txt', 'a')
        

        if config['factcheck']['save_files']:
            save_file(fake_facts, f'/home/ubuntu/thesis/source/factcheck/fake_facts.txt')
            #save_file(llm_revised_fake_facts, f'/home/ubuntu/thesis/source/factcheck/{config['model']['refinement_model']}_llm_revised_fake_facts.txt')
            save_file(gt_revised_fake_facts, f'/home/ubuntu/thesis/source/factcheck/gt_revised_fake_facts.txt')

            save_file(true_facts, f'/home/ubuntu/thesis/source/factcheck/true_facts.txt')
            #save_file(llm_revised_true_facts, f'/home/ubuntu/thesis/source/factcheck//{config['model']['refinement_model']}_llm_revised_true_facts.txt')
            save_file(gt_revised_true_facts, f'/home/ubuntu/thesis/source/factcheck/gt_revised_true_facts.txt')
    
    else:
        fake_facts = []
        with open('/home/ubuntu/thesis/source/factcheck/fake_facts.txt', 'r') as file:
            fake_facts = [line.strip() for line in file]
        llm_revised_fake_facts = []
        with open('/home/ubuntu/thesis/source/factcheck/llm_revised_fake_facts.txt', 'r') as file:
            llm_revised_fake_facts = [line.strip() for line in file]
        gt_revised_fake_facts = []
        with open('/home/ubuntu/thesis/source/factcheck/gt_revised_fake_facts.txt', 'r') as file:
            gt_revised_fake_facts = [line.strip() for line in file]

        true_facts = []
        with open('/home/ubuntu/thesis/source/factcheck/true_facts.txt', 'r') as file:
            true_facts = [line.strip() for line in file]
        llm_revised_true_facts = []
        with open('/home/ubuntu/thesis/source/factcheck/llm_revised_true_facts.txt', 'r') as file:
            llm_revised_true_facts = [line.strip() for line in file]
        gt_revised_true_facts = []
        with open('/home/ubuntu/thesis/source/factcheck/gt_revised_true_facts.txt', 'r') as file:
            gt_revised_true_facts = [line.strip() for line in file]



    ############### Metric EVALUATION #####################
    if os.path.exists('/home/ubuntu/thesis/source/factcheck/fake_evals.json'):
            with open('/home/ubuntu/thesis/source/factcheck/fake_evals.json', 'r') as file:
                fake_evals = json.load(file)    
    
    else:
        fake_evals = {
            "inclusions": 0,
            "conflicts": 0,
            "equivalences": 0,
            "wrong_gts": 0,
            "count": 0

        }  

    inclusions = fake_evals['inclusions']
    conflicts = fake_evals['conflicts']
    equivalences = fake_evals['equivalences']
    wrong_gts = fake_evals['wrong_gts']
    start_idx = fake_evals['count']

    llm_revised_fake_facts = []
    with open('/home/ubuntu/thesis/source/factcheck/_Ollama llama3.3_llm_revised_fake_facts.txt', 'r') as file:
        llm_revised_fake_facts = file.read().split('________________________________________________________________________________')
    llm_revised_fake_facts = [fact.strip() for fact in llm_revised_fake_facts if fact.strip()]    


    print("\n\nEvaluating Fake Facts...")
    for i in range(start_idx, len(fake_facts)):
        print(f"\nEvaluating {i+1}/{len(fake_facts)})")
        if are_semantically_equivalent(llm_revised_fake_facts[i], gt_revised_fake_facts[i], model=config['model']['checking_model']):
            fake_evals['equivalences'] += 1
            print("\nSEMANTICALLY EQUIVALENT!")
        elif is_semantically_contained(gt_revised_fake_facts[i], llm_revised_fake_facts[i], model=config['model']['checking_model']):
            fake_evals['inclusions'] += 1
            print("\nSEMANTICALLY INCLUDED!")
        elif are_semantically_conflicting(llm_revised_fake_facts[i], gt_revised_fake_facts[i], model=config['model']['checking_model']):
            fake_evals['conflicts'] += 1
            print("\nCONFLICT!")
            comparison = compare_correctness(llm_revised_fake_facts[i], gt_revised_fake_facts[i], model=config['model']['checking_model'])
            if comparison == 1:
                print("But actually GT is wrong!")
                fake_evals['wrong_gts'] += 1
            elif comparison == 2:
                print("Your method got it wrong! :(")
            else:
                print("Failed to compare the two head-to-head.")
            fake_evals['count'] += 1

        else: print("\nNOTHING DETECTED!")
        print(f"Original Fake Fact: {original_fake_facts[i]} \nLLM: {llm_revised_fake_facts[i]} \nGT: {gt_revised_fake_facts[i]}")
        save_file(fake_evals, '/home/ubuntu/thesis/source/factcheck/fake_evals.json')
        
    print("\nFake Facts:")
    print(f"Equivalences: {fake_evals['equivalences']}/{len(fake_facts)}")
    print(f"Inclusions: {fake_evals['inclusions']}/{len(fake_facts)}")
    print(f"Conflicts: {fake_evals['conflicts']}/{len(fake_facts)}")
    print(f"Inconclusive: {len(fake_facts)-fake_evals['equivalences']-fake_evals['inclusions']-fake_evals['conflicts']}/{len(fake_facts)}")
    print(f"Wrong ground truth refinements: {fake_evals['wrong_gts']}/{len(true_facts)}")
    save_file(fake_evals, '/home/ubuntu/thesis/source/factcheck/fake_evals.json')


    if os.path.exists('/home/ubuntu/thesis/source/factcheck/true_evals.json'):
            with open('/home/ubuntu/thesis/source/factcheck/true_evals.json', 'r') as file:
                true_evals = json.load(file)
    else:
        true_evals = {
            "inclusions": 0,
            "conflicts": 0,
            "equivalences": 0,
            "wrong_gts": 0,
            "count": 0

        }

    inclusions = true_evals['inclusions']
    conflicts = true_evals['conflicts']
    equivalences = true_evals['equivalences']
    wrong_gts = true_evals['wrong_gts']
    start_idx = true_evals['count']

    llm_revised_true_facts = []
    with open('/home/ubuntu/thesis/source/factcheck/Ollama llama3.3_llm_revised_true_facts.txt', 'r') as file:
        llm_revised_true_facts = file.read().split('________________________________________________________________________________')
    llm_revised_true_facts = [fact.strip() for fact in llm_revised_true_facts if fact.strip()]  
    
    print("\n\nEvaluating True Facts...")
    for i in range(start_idx, len(true_facts)):
        print(f"\nEvaluating {i+1}/{len(true_facts)})")
        if are_semantically_equivalent(llm_revised_true_facts[i], gt_revised_true_facts[i]):
            true_evals['equivalences'] += 1
            print("\nSEMANTICALLY EQUIVALENT!")
        elif is_semantically_contained(gt_revised_true_facts[i], llm_revised_true_facts[i]):
            true_evals['inclusios'] += 1
            print("\nSEMANTICALLY INCLUDED!")
        elif are_semantically_conflicting(llm_revised_true_facts[i], gt_revised_true_facts[i]):
            true_evals['conflicts'] += 1
            print("\nCONFLICT!")
            comparison = compare_correctness(llm_revised_true_facts[i], gt_revised_true_facts[i], model=config['model']['checking_model'])
            if comparison == 1:    
                print("But actually GT is wrong!")
                true_evals['wrong_gts'] += 1
            elif comparison == 2:
                print("Your method got it wrong! :(")
            else:
                print("Failed to compare the two head-to-head.")
        else: print("\nNOTHING DETECTED!")
        print(f"Original True Fact: {original_true_facts[i]} \nLLM: {llm_revised_true_facts[i]} \nGT: {gt_revised_true_facts[i]}")
        save_file(fake_evals, '/home/ubuntu/thesis/source/factcheck/true_evals.json')
        

    print("\nTrue Facts:")
    print(f"Equivalences: {true_evals['equivalences']}/{len(true_facts)}")
    print(f"Inclusions: {true_evals['inclusions']}/{len(true_facts)}")
    print(f"Conflicts: {true_evals['conflicts']}/{len(true_facts)}")
    print(f"Inconclusive: {len(true_facts)-true_evals['equivalences']-true_evals['inclusions']-true_evals['conflicts']}/{len(true_facts)}")
    print(f"Wrong ground truth refinements: {true_evals['wrong_gts']}/{len(true_facts)}")
    save_file(true_evals, '/home/ubuntu/thesis/source/factcheck/true_evals.json')

if __name__ == "__main__":
    #main()
    config = load_config()
    with open(f'/home/ubuntu/thesis/source/factcheck/{config['model']['refinement_model']}_output_log.txt', 'w') as log_file:
        dual_stdout = DualStdout(log_file)
        original_stdout = sys.stdout
        sys.stdout = dual_stdout

        main()
        sys.stdout = original_stdout  # Reset standard output to its original value.
    print(f"Log file '{config['model']['refinement_model']}_output_log.txt' created.")
