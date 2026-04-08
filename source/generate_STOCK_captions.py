from google import genai
from google.genai import types
import os
import PIL.Image
import json
from helpers import(
    load_config,
    generate_prompt_for_baseline,
    save_file,
    get_response
)
from claude_api import(
    get_claude_response,
    get_claude_image_response
)


import openai
from PIL import Image
import base64
# Load model directly
from transformers import AutoModel, AutoTokenizer
import torch


def get_vlm_response(model_name, prompt, image_paths):
    if type(image_paths) is not list: image_paths = [image_paths]
    if model_name == "gemini-2.0-flash":
        with open("/home/ubuntu/thesis/.credentials/google", "r") as file:
            google_api_key = file.read().strip()
        client = genai.Client(api_key=google_api_key)

        # Open and store all image objects
        images = [Image.open(path) for path in image_paths]

        # Construct content with prompt followed by all images
        contents = [prompt] + images

        response = client.models.generate_content(
            model=model_name,
            contents=contents
        )
        return response.text
    
    elif model_name == "OpenAI GPT-4o":
        model_name = "gpt-4o"
        # Read OpenAI API key from credentials file
        with open("/home/ubuntu/thesis/.credentials/openai", "r") as file:
            openai_api_key = file.read().strip()
        
        # Initialize OpenAI client
        client = openai.OpenAI(api_key=openai_api_key)
        
        # Prepare message content
        message_content = [
            {"type": "text", "text": prompt}
        ]
        
        # Add images to the message content
        for path in image_paths:
            try:
                # Encode image to base64
                with open(path, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                
                message_content.append({
                    "type": "image_url", 
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                })
            except FileNotFoundError:
                print(f"Error: Image not found at {path}")
                return None
            except Exception as e:
                print(f"Error processing image {path}: {e}")
                return None
        
        # Make API call to GPT-4o
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": message_content
                }
            ],
            max_tokens=300
        )
        
        # Return the response text
        return response.choices[0].message.content

    elif model_name == "gemini-2.5-pro-preview-05-06":
        client = genai.Client(
            vertexai=True,
            project="ts-captioning",
            location="us-central1",
        )

        with open(image_paths[0], "rb") as img_file:
            img_bytes = img_file.read()
        msg1_image1 = types.Part.from_bytes(
            data=img_bytes,
            mime_type="image/jpeg",
        )
        
        #print("\n\n", type(msg1_image1), "\n\n")
        msg1_text1 = types.Part.from_text(text=prompt)

        model = model_name
        contents = [
            types.Content(
            role="user",
            parts=[
                msg1_image1,
                msg1_text1
            ]
            ),
        ]

        generate_content_config = types.GenerateContentConfig(
            temperature = 0.3,
            top_p = 0.95,
            seed = 0,
            max_output_tokens = 65535,
            safety_settings = [types.SafetySetting(
            category="HARM_CATEGORY_HATE_SPEECH",
            threshold="OFF"
            ),types.SafetySetting(
            category="HARM_CATEGORY_DANGEROUS_CONTENT",
            threshold="OFF"
            ),types.SafetySetting(
            category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
            threshold="OFF"
            ),types.SafetySetting(
            category="HARM_CATEGORY_HARASSMENT",
            threshold="OFF"
            )],
        )

        response = client.models.generate_content(
            model = model,
            contents = contents,
            config = generate_content_config,
        )

        return response.text
    
    else:
        print(f"Unsupported model: {model_name}")
        return None


def main():
    config = load_config()
    use_img_input = config['eval']['use_img_input']
    
    """image_path = "/home/ubuntu/thesis/data/samples/plots/agriculture_0.jpeg"
    prompt = "Please describe this time series about the yearly Aggregated input index (2015=100) in Senegal, in the context of agriculture. Starting from 2008 and ending in 2013. Answer in a single concise paragraph, without formatting."
    
    print(get_vlm_response(model_name="gemini-2.0-flash", prompt=prompt, image_path=image_path))"""
    
    
    model_name = config['STOCK']['generative_model']
    if "gemini" not in model_name.lower() and "gpt" not in model_name.lower() and "claude" not in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained("OpenGVLab/InternVL2_5-8B", trust_remote_code=True)
        model = AutoModel.from_pretrained("OpenGVLab/InternVL2_5-8B", trust_remote_code=True, torch_dtype="auto")
        model.eval()
                    
    ts_folder_path = config['STOCK']['ts_path']
    save_folder_path =  config['STOCK']['save_root_path'] +"/" + model_name #"/home/ubuntu/thesis/data/stock_data/synth-captions" 
    
    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)
    
    done_caption_ids = [filename.split(".")[0].split("_")[1] for filename in os.listdir(save_folder_path)]
    
    ids = list(range(0, 999))
    
    ids = [id for id in ids if str(id) not in done_caption_ids]
    
    print(f"\n{model_name} {"with" if use_img_input else "without"} image input: {len(ids)} captions yet to be generated.\n\n")
    
    
    for i, id in enumerate(ids):
        ts_path = os.path.join(ts_folder_path, f"ts_{id}.txt")
        
        with open(ts_path, "r") as ts_file:
            ts = ts_file.read().strip()
        
        prompt = f"""
Your task is to analyze the time series data and describe its key features.

Your response must follow these rules exactly:

Provide three short, descriptive phrases.
Each phrase must describe a single key feature.
Phrases should be terse, like captions, not full sentences. Do not explicitly include numbers.

Here are some examples for reference:

Example 1:
[65.0 73.0 72.0 69.0 76.0 95.0 98.0 104.0 101.0 104.0 100.0 96.0]
starts with a shallow increase
rise in the middle
slight trough in the ending part

Example 2:
[2.0 53.0 42.0 37.0 55.0 53.0 6.0 5.0 13.0 13.0 11.0 12.0 13.0]
lowest value in the beginning
dip in the middle
stable at the ending part

Example 3:
[48.0 56.0 46.0 41.0 55.0 69.0 68.0 79.0 86.0 70.0 62.0 58.0]
a slight increase in the beginning
peak after the middle
declines at end

Example 4:
[20.0 18.0 19.0 20.0 16.0 14.0 15.0 24.0 25.0 37.0 33.0 39.0]
stays level initially
dip around the middle
ends at a higher value than the beginning

Example 5:
[18.0 19.0 21.0 21.0 25.0 30.0 20.0 14.0]
steady increase from the beginning
biggest rise in the second half
sharp drop by the end

Here are some other examples of features to say:
small valley in the middle
rises in waves
growth from the middle onwards
maximum value after a sharp incline
plot stays mostly flat for length
line experiences a trough in the halfway point
two troughs around the middle
end is much higher than the start 
has a horizontal trend
rises twice
first two-thirds is increasing


You do not have to use the same language shown in these examples, I only care about the correctness of the descriptions. Do not add any explanations or additional text, I just want the description in three to four lines. Now, here's the time series for you to describe:
[{ts}]
        """

        if "claude" in model_name: 
            model = "bedrock/us.anthropic.claude-3-haiku-20240307-v1:0"
            generated_caption = get_claude_response(prompt, model=model)
        elif "gemini" in model_name.lower() or "gpt" in model_name.lower():
            if model_name == "gemini-2.0-flash" or model_name == "Google Gemini-2.0-flash": 
                model_name = "Google Gemini-2.0-Flash"
                
            generated_caption = get_response(prompt=prompt, model=model_name)
            
        else: # open-source model
            inputs = tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                generated_caption = model(**inputs)

        
        #print(generated_caption)
        save_file(data=generated_caption, filepath=save_folder_path+f"/caption_{id}.txt")
        
        if i % 50 == 0 and i != 0:
            print(f"\n{i}/{len(ids)} Done.")
        
        
    
if __name__ == "__main__":
    main()
    

