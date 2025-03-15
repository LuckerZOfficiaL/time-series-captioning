import json
from helpers import (
    save_file,
    split_facts_by_time,
    load_config
)

"""
BIN_YEARS = 10 # how long is a single period in years?
BANK_PATH = "/home/ubuntu/thesis/data/fact bank/all_facts_no_common_sense.txt"
SAVE_PATH = f"/home/ubuntu/thesis/data/fact bank/by period/{BIN_YEARS}/all_facts_by_{BIN_YEARS}years.json"
"""

def main():
    config = load_config()
    bin_years = config['bank']['bin_years']
    bank_path = config['path']['all_facts_no_common_sense_path']
    save_path = f"{config['path']['all_facts_by_period_folder_path']}/{bin_years}/all_facts_by_{bin_years}years.json"


    with open(bank_path, 'r') as file:
        fact_bank = file.read()
    facts_list = fact_bank.split("\n")
    
    time_periods = split_facts_by_time(facts_list, bin_years = bin_years)
    save_file(time_periods, save_path)

    print(f"\nSuccess: Split facts into time periods of {bin_years} years.")
            

if __name__ == "__main__":
    main()