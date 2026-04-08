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
import os

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
    
    
    model_name = config['eval']['evaluated_model']
    
    ts_folder_path = "/home/ubuntu/thesis/data/samples/new_samples_no_overlap/test/time series"
    metadata_folder_path = "/home/ubuntu/thesis/data/samples/new_samples_no_overlap/test/metadata"
    image_folder_path = "/home/ubuntu/thesis/data/samples/new_samples_no_overlap/test/plots"
    save_folder_path = f"/home/ubuntu/thesis/data/samples/new_samples_no_overlap/generated_captions/new_prompt/{model_name}{"" if use_img_input else "_text"}"
    
    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)
    
    done_caption_ids = [filename.split(".")[0] for filename in os.listdir(save_folder_path)]
    
    filenames = os.listdir(image_folder_path)
    filenames = [filename for filename in filenames if filename.split(".")[0] not in done_caption_ids]
    #filenames = [filename.replace("_test", "") for filename in filenames]
    
    print(f"\n{model_name} {"with" if use_img_input else "without"} image input: {len(filenames)} captions yet to be generated.\n\n")
    
    
    for i, filename in enumerate(filenames):
        image_path = os.path.join(image_folder_path, filename)
        metadata_path = os.path.join(metadata_folder_path, filename[:-4]+"json")
        ts_path = os.path.join(ts_folder_path, filename[:-4]+"txt")
        dataset_name = filename.split("_")[0]
        
        #if dataset_name not in ['crime', 'demography', 'agriculture']:
        #    continue
        
        with open(ts_path, "r") as ts_file:
            ts = [float(line.strip()) for line in ts_file]
            
        with open(metadata_path, "r") as metadata_file:
            metadata = json.load(metadata_file)
        
        prompt = generate_prompt_for_baseline(dataset_name=dataset_name, metadata=metadata, ts=ts)
        
        
        if use_img_input:
            prompt = prompt + "\nI have attached a line plot of the time series to support you."
            if "claude-3-haiku" in model_name:
                generated_caption = get_claude_image_response(image_path, prompt, model="bedrock/us.anthropic.claude-3-haiku-20240307-v1:0")
            else:
                if model_name == "GPT-4o" or model_name == "gpt-4o" or model_name == "OpenAI GPT-4o":
                    model_name = "OpenAI GPT-4o"
                generated_caption = get_vlm_response(model_name=model_name, prompt=prompt, image_paths=image_path)
            
        else:
            if "claude-3-haiku" in model_name:
                generated_caption = get_claude_response(prompt, model="bedrock/us.anthropic.claude-3-haiku-20240307-v1:0")
            elif "claude" in model_name:
                generated_caption = get_claude_response(prompt, model=model_name)
            else:
                if model_name == "gemini-2.0-flash":
                    model = "Google Gemini-2.0-Flash"
                elif model_name == "GPT-4o" or model_name == "gpt-4o" or model_name == "OpenAI GPT-4o":
                    model = "OpenAI GPT-4o"
                elif model_name == "gemini-3-pro-preview":
                    model = "Google Gemini-3-Pro-Preview"
                elif model_name == "gpt-5.1":
                    model = "OpenAI GPT-5.1"
                elif model_name == "gpt-5-mini":
                    model = "gpt-5-mini"
                generated_caption = get_response(prompt=prompt, model=model)
                #print(generated_caption)
                #exit()
        
        #print(generated_caption)
        save_file(data=generated_caption, filepath=save_folder_path+"/"+filename[:-4]+"txt")
        
        if i % 50 == 0 and i != 0:
            print(f"\n{i}/{len(filenames)} Done.")
        
        
    
if __name__ == "__main__":
    main()
    

