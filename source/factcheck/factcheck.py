import json
import sys
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
        
        #print(facts[1].keys())
        
        print(f"{i}/{len(facts)}")
        exit()

        for fact in facts:
            print(fact['response_factuality'])

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
            elif facts[i]['response_factuality'] == True:
                true_facts.append(facts[i]['response'])
                gt_revised_true_facts.append(facts[i]['revised_response'])
            else: # the factuality is non defined
                pass
            #if i+1 == 3 : break
        
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
        

        save_file(fake_facts, '/home/ubuntu/thesis/source/factcheck/fake_facts.txt')
        save_file(llm_revised_fake_facts, '/home/ubuntu/thesis/source/factcheck/llm_revised_fake_facts.txt')
        save_file(gt_revised_fake_facts, '/home/ubuntu/thesis/source/factcheck/gt_revised_fake_facts.txt')

        save_file(true_facts, '/home/ubuntu/thesis/source/factcheck/true_facts.txt')
        save_file(llm_revised_true_facts, '/home/ubuntu/thesis/source/factcheck/llm_revised_true_facts.txt')
        save_file(gt_revised_true_facts, '/home/ubuntu/thesis/source/factcheck/gt_revised_true_facts.txt')
    
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
    inclusions = 0
    conflicts = 0
    equivalences = 0
    wrong_gts = 0
    i = 1
    print("\n\nEvaluating Fake Facts...")
    for original_fake_fact, llm_revised_fake_fact, gt_revised_fake_fact in zip(fake_facts, llm_revised_fake_facts, gt_revised_fake_facts):
        print(f"\nEvaluating {i}/{len(fake_facts)})")
        i += 1
        if are_semantically_equivalent(llm_revised_fake_fact, gt_revised_fake_fact):
            equivalences += 1
            print("\nSEMANTICALLY EQUIVALENT!")
        elif is_semantically_contained(gt_revised_fake_fact, llm_revised_fake_fact):
            inclusions += 1
            print("\nSEMANTICALLY INCLUDED!")
        elif are_semantically_conflicting(llm_revised_fake_fact, gt_revised_fake_fact):
            conflicts += 1
            print("\nCONFLICT!")
            comparison = compare_correctness(llm_revised_fake_fact, gt_revised_fake_fact, model=config['model']['checking_model'])
            if comparison == 1:
                print("But actually GT is wrong!")
                wrong_gts += 1
            elif comparison == 2:
                print("Your method got it wrong! :(")
            else:
                print("Failed to compare the two head-to-head.")

        else: print("\nNOTHING DETECTED!")
        print(f"Original Fake Fact: {original_fake_fact} \nLLM: {llm_revised_fake_fact} \nGT: {gt_revised_fake_fact}")
        
    print("\nFake Facts:")
    print(f"Equivalences: {equivalences}/{len(fake_facts)}")
    print(f"Inclusions: {inclusions}/{len(fake_facts)}")
    print(f"Conflicts: {conflicts}/{len(fake_facts)}")
    print(f"Inconclusive: {len(fake_facts)-equivalences-inclusions-conflicts}/{len(fake_facts)}")
    print(f"Wrong ground truth refinements: {wrong_gts}/{len(true_facts)}")



    inclusions = 0
    conflicts = 0
    equivalences = 0
    wrong_gts = 0
    i = 1
    print("\n\nEvaluating True Facts...")
    for original_true_fact, llm_revised_true_fact, gt_revised_true_fact in zip(true_facts, llm_revised_true_facts, gt_revised_true_facts):
        print(f"\nEvaluating {i}/{len(true_facts)})")
        i += 1
        if are_semantically_equivalent(llm_revised_true_fact, gt_revised_true_fact):
            equivalences += 1
            print("\nSEMANTICALLY EQUIVALENT!")
        elif is_semantically_contained(gt_revised_true_fact, llm_revised_true_fact):
            inclusions += 1
            print("\nSEMANTICALLY INCLUDED!")
        elif are_semantically_conflicting(llm_revised_true_fact, gt_revised_true_fact):
            conflicts += 1
            print("\nCONFLICT!")
            comparison = compare_correctness(llm_revised_true_fact, gt_revised_true_fact, model=config['model']['checking_model'])
            if comparison == 1:    
                print("But actually GT is wrong!")
                wrong_gts += 1
            elif comparison == 2:
                print("Your method got it wrong! :(")
            else:
                print("Failed to compare the two head-to-head.")
        else: print("\nNOTHING DETECTED!")
        print(f"Original True Fact: {original_true_fact} \nLLM: {llm_revised_true_fact} \nGT: {gt_revised_true_fact}")
        

    print("\nTrue Facts:")
    print(f"Equivalences: {equivalences}/{len(true_facts)}")
    print(f"Inclusions: {inclusions}/{len(true_facts)}")
    print(f"Conflicts: {conflicts}/{len(true_facts)}")
    print(f"Inconclusive: {len(true_facts)-equivalences-inclusions-conflicts}/{len(true_facts)}")
    print(f"Wrong ground truth refinements: {wrong_gts}/{len(true_facts)}")

if __name__ == "__main__":
    #main()
    with open('/home/ubuntu/thesis/source/factcheck/output_log.txt', 'w') as log_file:
        dual_stdout = DualStdout(log_file)
        original_stdout = sys.stdout
        sys.stdout = dual_stdout

        main()
        sys.stdout = original_stdout  # Reset standard output to its original value.
    print("Log file 'output_log.txt' created.")
