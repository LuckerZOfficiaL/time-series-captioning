import json
import os
from helpers import (
    save_file,
    remove_common_sense
)

MODEL = "Google Gemini-2.0-Flash" #"OpenAI GPT-4o" #"Gemini-2.0-Flash"
BANK_PATH = "/home/ubuntu/thesis/data/fact bank/all_facts.txt"
SAVE_PATH = "/home/ubuntu/thesis/data/fact bank/all_facts_no_common_sense.txt"

def main():
    with open(BANK_PATH, 'r') as file:
        fact_bank = file.read()
    fact_list = fact_bank.split("\n")
    
    new_fact_list = remove_common_sense(fact_list, SAVE_PATH, model=MODEL)
    save_file(new_fact_list, SAVE_PATH)
    print("\nSuccess: removed common sense")
            

if __name__ == "__main__":
    main()