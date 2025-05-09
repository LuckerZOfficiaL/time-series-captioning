from google import genai
from google.genai import types
import os
import PIL.Image as Image
import json
import os
from litellm import completion
from helpers import(
    load_config,
    generate_prompt_for_baseline,
    save_file,
    get_response
)
from claude_api import(
    get_claude_response,
    get_claude_image_response,
    encode_image
)


def get_vlm_response_with_filenames(model_name, prompt, image_paths):
    if model_name.lower() == "google gemini-2.0-flash": # this is because in the config file, gemini is named this way
        model_name = "gemini-2.0-flash"
        
    if model_name == "gemini-2.0-flash":
        with open("/home/ubuntu/thesis/.credentials/google", "r") as file:
            google_api_key = file.read().strip()
        client = genai.Client(api_key=google_api_key)

        contents = [prompt]
        images = []
        for path in image_paths:
            try:
                img = Image.open(path)
                images.append(img)
                filename = path.split('/')[-1]  # Extract filename from path
                contents.append(f"This is the content of the image file: {filename}")
                contents.append(img)
            except FileNotFoundError:
                print(f"Error: Image not found at {path}")
                return None
            except Exception as e:
                print(f"Error opening image {path}: {e}")
                return None

        response = client.models.generate_content(
            model=model_name,
            contents=contents
        )
        return response.text


def get_claude_response_with_filenames(prompt, image_paths, model="bedrock/us.anthropic.claude-3-haiku-20240307-v1:0"):
    """
    Get responses from Claude for a prompt with multiple images, referencing their filenames.

    Args:
        model_name (str): Name of the model to use (e.g., "bedrock/anthropic.claude-3-5-haiku-20241022-v1:0")
        prompt (str): Prompt to send to the model.
        image_paths (list): List of paths to the image files.
        model (str): The specific Claude model identifier.

    Returns:
        str: The model's response.
    """

    contents = [{"type": "text", "text": prompt}]
    for img_path in image_paths:
        base64_image = encode_image(img_path)
        if base64_image:
            filename = img_path.split('/')[-1]
            contents.append({"type": "text", "text": f"This is the content of the image file: {filename}"})
            contents.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            })
        else:
            return None  # Or handle the error as needed

    response = completion(
        model=model,
        messages=[
            {
                "role": "user",
                "content": contents
            }
        ]
    )

    #print(response)
    try:
        return response.choices[0].message.content
    except (AttributeError, IndexError, KeyError):
        print("Error: Could not extract text from Claude's response.")
        return None
    



def main():
    config = load_config()
    model = config['model']['qa_model']
    
    results_path = f"/home/ubuntu/thesis/data/samples/new samples no overlap/hard_questions_small/plot_retrieval_same_domain/{model}.json"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    answers = []
        
    questions_path = "/home/ubuntu/thesis/data/samples/new samples no overlap/hard_questions_small/plot_retrieval_same_domain_questions.json"
    with open(questions_path, "r") as file:
        questions = json.load(file)
        
    correct = 0
    tot = len(questions)
    
    print(f"Evaluating {model} on Plot Matching...")
    for i, question in enumerate(questions):
        prompt = f"""{question["Question"]} 
        
        Pick one of the following:
        {question['Option 1']}
        {question['Option 2']}
        {question['Option 3']}
        {question['Option 4']}
               
        Return your answer with the filename of the image and nothing else. Do not add any extra text or formatting.
        """ 
        img_files = [question['Option 1'], question['Option 2'], question['Option 3'], question['Option 4']]
        image_paths = [f"/home/ubuntu/thesis/data/samples/new samples no overlap/test/plots/{img_file}" for img_file in img_files]
        
        #prompt = "Describe these images briefly, for each image file."
        if model == "claude-3-haiku":
            response = get_claude_response_with_filenames(prompt=prompt, image_paths=image_paths)
        else:
            response = get_vlm_response_with_filenames(model, prompt, image_paths).lower()
        #print("\nPrompt: ", prompt)
        #print("Response: ", response)
        if question['Ground Truth'] in response and all(option not in response for option in img_files if option != question['Ground Truth']):
            is_correct = True
            correct += 1
        else:
            is_correct = False

        answers.append({question['Question ID']: {"response": response, "correct": is_correct}})
        
        #if i % 3 == 0 and i != 0:
        print(f"Done for {i+1}, correct: {correct}/{i+1}")
        with open(results_path, "w") as outfile:
            json.dump(answers, outfile, indent=4)
            
        #if i == 1: break
        
    answers.append({"overall accuracy": correct/tot})
    with open(results_path, "w") as outfile:
            json.dump(answers, outfile, indent=4)
        
    
if __name__ == "__main__":
    main()
    
    