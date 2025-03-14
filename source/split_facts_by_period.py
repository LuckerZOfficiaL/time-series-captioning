import json
from helpers import (
    save_file,
    split_facts_by_time
)

BIN_YEARS = 10 # how long is a single period in years?
BANK_PATH = "/home/ubuntu/thesis/data/fact bank/all_facts.txt"
SAVE_PATH = f"/home/ubuntu/thesis/data/fact bank/by period/{BIN_YEARS}/all_facts_by_{BIN_YEARS}years.json"

def main():
    with open(BANK_PATH, 'r') as file:
        fact_bank = file.read()
    facts_list = fact_bank.split("\n")
    
    time_periods = split_facts_by_time(facts_list, bin_years = BIN_YEARS)
    save_file(time_periods, SAVE_PATH)

    print("\nSuccess: Split facts into time periods.")
            

if __name__ == "__main__":
    main()