from helpers import(
    get_response,
    load_config
)
from claude_api import get_claude_response
import json
import os
import re


_ALLOWED_ANSWERS = {"a", "b", "c", "d", "true", "false"}

def _parse_start(s: str):
    # Grab the first contiguous token after leading whitespace
    m = re.match(r"\s*(\S+)", s)
    if not m:
        print("String is empty or whitespace only.")
        return None
    token = m.group(1).lower()
    if token not in _ALLOWED_ANSWERS:
        print(f"Unrecognised answer '{m.group(1)}' in: {s!r}")
        return None
    return token.title()
        

def _attempt_parse_str(s):
    s = s.lower()
    assert not (("false" in s) and ("true" in s))
    if "false" in s:
        return False
    if "true" in s:
        return True
    str_reformat = s.replace('.', '').replace('"', '').replace("'", '').split(' ')
    for x in str_reformat:
        if x in _ALLOWED_ANSWERS:
            return _parse_start(x)
    print(f"INVALID ANSWER:\n {s}")

    return ""

def extract_choice(json_str):
    orig_str = json_str
    json_str = json_str.replace('```', '').replace('json', '')
    json_str = re.sub(r'("answer"\s*:\s*)([A-Za-z])', r'\1"\2"', json_str)
    try:
        data = json.loads(json_str)
    except:
        return _attempt_parse_str(json_str)
    
    try:
        answer = data.get("answer", "").strip()
    except AttributeError:
        # handle LLM answering only 'true' or 'false'
        if type(data) == bool:
            return str(data)
        return _attempt_parse_str(data)
        import pdb; pdb.set_trace()

    return _parse_start(answer)




def main():
    config = load_config()
    task = config['new_tasks']['qa_task']
    
    questions_path = f"/home/ubuntu/thesis/data/samples/new samples no overlap/hard_questions_small/{task}/tasks.json"
    
    evaluated_model = config['model']['qa_model']
    
    results_path = f"/home/ubuntu/thesis/data/samples/new samples no overlap/hard_questions_small/{task}/{evaluated_model}.json"
    
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    
    if os.path.exists(results_path):
        with open(results_path, "r") as file:
            results = json.load(file)
    else:
        results = {}
    
    with open(questions_path, "r") as file:
        questions = json.load(file)
        
    num_questions = len(questions)
    
    print(f"Evaluating {evaluated_model} on Q&A task {task}...")
    if "prompt_no_image" not in questions[0].keys():
        prompt_key = "prompt"
    else:
        prompt_key = "prompt_no_image" 
        
    for i, question in enumerate(questions):
        if question['ts_name'] not in results.keys():
            if evaluated_model == "claude-3-haiku":
                response = get_claude_response(prompt=question[prompt_key])
            else:
                response = get_response(prompt=question[prompt_key], model=evaluated_model, temperature=0.2)
            response = response.lower()
            
            if "true" in response:
                answer = "True"
            elif "false" in response:
                answer = "False"
            else:
                #answer = next((char for char in "ABCD" if char in response), None)
                answer = extract_choice(response)
                
            
            #answer = extract_answer(response)
            results[question['ts_name']] = {}
            results[question['ts_name']]['answer'] = answer
            results[question['ts_name']]['ground_truth'] = str(question["ground_truth"])
            results[question['ts_name']]['correct'] = (answer == str(question["ground_truth"]))
            
        with open(results_path, "w") as file:
            json.dump(results, file, indent=4)
        
        if i % 100 == 0 and i != 0:
            print(f"Done for {i}/{num_questions}")
            
            
    if os.path.exists(results_path):
        with open(results_path, "r") as file:
            results = json.load(file)
            
    correct_count = sum(1 for result in results.values() if result['correct'])
    accuracy = correct_count / num_questions
    results['overall'] = {}
    results['overall']['accuracy'] = accuracy
    
    print(f"Average accuracy: {accuracy:.2%}")
    
    with open(results_path, "w") as file:
            json.dump(results, file, indent=4)
    print("Done.")
            
    

if __name__ == "__main__":
    main()