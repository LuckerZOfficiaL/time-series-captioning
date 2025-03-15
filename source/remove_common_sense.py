import json
import os
from helpers import (
    save_file,
    remove_common_sense,
    load_config
)

"""MODEL = "Google Gemini-2.0-Flash" #"OpenAI GPT-4o" #"Gemini-2.0-Flash"
BANK_PATH = "/home/ubuntu/thesis/data/fact bank/all_facts.txt"
SAVE_PATH = "/home/ubuntu/thesis/data/fact bank/all_facts_no_common_sense.txt"
BATCH_SIZE = 8 # how many facts to present to the LLM in each prompt for removing common sense?"""

def main():
    config = load_config()
    model = config['model']['remove_common_sense_model']
    bank_path = config['path']['all_facts_path']
    save_path = config['path']['all_facts_no_common_sense_path']
    batch_size = config['refinement']['batch_size']


    with open(bank_path, 'r') as file:
        fact_bank = file.read()
    fact_list = fact_bank.split("\n")

    new_fact_list = remove_common_sense(fact_list, save_path, model=model, batch_size=batch_size)
    save_file(new_fact_list, save_path)
    print(f"\nSuccess: removed common sense: {len(fact_list)} - {len(fact_list)-len(new_fact_list)} = {len(new_fact_list)} facts remaining.")
            

if __name__ == "__main__":
    main()