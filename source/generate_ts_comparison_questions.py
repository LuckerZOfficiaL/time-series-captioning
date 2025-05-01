import os
import random
import json

from sympy import EX

from helpers import(
    load_config,
    create_volatility_question,
    create_mean_question,
    create_same_phenomenon_question,
    create_peak_earlier_question,
    create_bottom_earlier_question,
    create_amplitude_question,
    read_txt_to_num_list
)


def main():
    config = load_config()
    
    gt_ts_folder_path = config['path']['gt_ts_folder_path']
    comparison_questions_folder_path = config['path']['comparison_questions_folder_path']
    num_questions = config['new_tasks']['comparison_questions_per_subtask']
    subtasks = config['new_tasks']['ts_comparison_tasks']
    
    
    for subtask in subtasks:
        prompt_save_folder = comparison_questions_folder_path + f"/{subtask}/prompts"
        answer_save_folder = comparison_questions_folder_path + f"/{subtask}/ground truth"
        os.makedirs(prompt_save_folder, exist_ok=True)
        os.makedirs(answer_save_folder, exist_ok=True)
        
        print(f"{num_questions} {subtask} questions to be generated, sampling GT time series from {gt_ts_folder_path}.")

        if subtask in ["volatility", "mean", "same_phenomenon", "peak_earlier", "bottom_earlier", "amplitude"]:        
            for i in range(num_questions):
                #if i % 100 == 0 and i != 0:
                #    print(f"Generating {subtask} question {i}/{num_questions}...")
                
                while True:
                    try:
                        ts1_filename = random.choice(os.listdir(gt_ts_folder_path))
                        ts1_filepath = os.path.join(gt_ts_folder_path, ts1_filename)
                        
                            
                        ts2_filename = random.choice(os.listdir(gt_ts_folder_path))
                        ts2_filepath = os.path.join(gt_ts_folder_path, ts2_filename)
                        ts2 = read_txt_to_num_list(ts2_filepath)
                
                
                        if subtask == "volatility":                   
                                    create_volatility_question(ts_path1=ts1_filepath, ts2=ts2, 
                                                prompt_save_folder=prompt_save_folder,
                                                answer_save_folder=answer_save_folder)

                                    
                        elif subtask == "mean":
                                    create_mean_question(ts_path1=ts1_filepath, ts2=ts2, 
                                                prompt_save_folder=prompt_save_folder,
                                                answer_save_folder=answer_save_folder)
                                    
                        elif subtask == "peak_earlier":
                                    create_peak_earlier_question(ts_path1=ts1_filepath, ts2=ts2, 
                                                prompt_save_folder=prompt_save_folder,
                                                answer_save_folder=answer_save_folder)
                                    
                        elif subtask == "bottom_earlier":
                                    create_bottom_earlier_question(ts_path1=ts1_filepath, ts2=ts2, 
                                                prompt_save_folder=prompt_save_folder,
                                                answer_save_folder=answer_save_folder)

                                    
                        elif subtask == "amplitude":
                                    create_amplitude_question(ts_path1=ts1_filepath, ts2=ts2, 
                                                prompt_save_folder=prompt_save_folder,
                                                answer_save_folder=answer_save_folder)
                        
                        elif subtask == "same_phenomenon":
                                #if i % 100 == 0 and i != 0:
                                #    print(f"Generating {subtask} question {i}/{num_questions}...")
                                same_phenom = random.choice([True, False])
                                
                                ts1_filename = random.choice(os.listdir(gt_ts_folder_path))
                                ts1_filepath = os.path.join(gt_ts_folder_path, ts1_filename)
                                
                                    
                                if not same_phenom:
                                    ts2_filename = random.choice(os.listdir(gt_ts_folder_path))
                                    ts2_filepath = os.path.join(gt_ts_folder_path, ts2_filename)
                                    ts2 = read_txt_to_num_list(ts2_filepath)
                                else:
                                    ts2 = None
                                create_same_phenomenon_question(ts_path1=ts1_filepath, ts2=ts2, 
                                                                same_phenom=same_phenom,
                                                                prompt_save_folder=prompt_save_folder,
                                                                answer_save_folder=answer_save_folder)    
                        
                        break
                    except Exception as e:
                        print(e)       
    
    


if __name__ == "__main__":
    main()