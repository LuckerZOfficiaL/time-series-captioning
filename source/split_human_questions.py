import json
import csv
import pandas as pd



def main():
    with open('/home/ubuntu/thesis/data/samples/new samples no overlap/hard_questions_small/human_eval.json', 'r') as file:
        data = json.load(file)
    grouped_data = {}
    for item in data:
        question_type = item.get("Question Type")
        if question_type not in grouped_data:
            grouped_data[question_type] = []
        grouped_data[question_type].append(item)
    
    for question_type, questions in grouped_data.items():
        output_path = f'/home/ubuntu/thesis/data/samples/new samples no overlap/hard_questions_small/{question_type}_questions.json'
        with open(output_path, 'w') as output_file:
            json.dump(questions, output_file, indent=4)
            
    for question_type, questions in grouped_data.items():
        output_csv_path = f'/home/ubuntu/thesis/data/samples/new samples no overlap/hard_questions_small/{question_type}_questions.csv'
        with open(output_csv_path, 'w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=questions[0].keys())
            writer.writeheader()
            writer.writerows(questions)
            
    
    volatility_questions_path = '/home/ubuntu/thesis/data/samples/new samples no overlap/hard_questions_small/volatility_questions.csv'
    volatility_df = pd.read_csv(volatility_questions_path)
    print(volatility_df.head())
    
    
    #df = pd.read_csv('/home/ubuntu/thesis/data/samples/new samples no overlap/hard_questions_small/plot_retrieval_same_domain_questions.csv')
    
    #img_folder_path = "/home/ubuntu/thesis/data/samples/new samples no overlap/test/plots"
    
    #grouped = {question_type: group for question_type, group in df.groupby('Question Type')}
    #for question_type, group_df in grouped.items():
    #    output_path = f'/home/ubuntu/thesis/data/samples/new samples no overlap/hard_questions_small/{question_type}_questions.csv'
    #    group_df.to_csv(output_path, index=False)
    
    
    """options_list = []
    for _, row in df.iterrows():
        options_list.extend([row['Option 1'], row['Option 2'], row['Option 3'], row['Option 4']])
    images = set(options_list)
    
    output_folder = "/home/ubuntu/thesis/data/samples/new samples no overlap/hard_questions_small/plots"
    os.makedirs(output_folder, exist_ok=True)

    for image in images:
        src_path = os.path.join(img_folder_path, image)
        dest_path = os.path.join(output_folder, image)
        if os.path.exists(src_path):
            shutil.copy(src_path, dest_path)"""
            
           
    
    
if __name__ == "__main__":
    main()