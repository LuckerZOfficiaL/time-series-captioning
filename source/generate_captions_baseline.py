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


def get_vlm_response(model_name, prompt, image_path):
    if model_name == "gemini-2.0-flash":
        with open("/home/ubuntu/thesis/.credentials/google", "r") as file:
              google_api_key = file.read().strip()
        client = genai.Client(api_key=google_api_key)
          
        image = PIL.Image.open(image_path)
        
        response = client.models.generate_content(
            model=model_name,
            contents=[prompt, image])
        return response.text


def main():
    config = load_config()
    use_img_input = config['eval']['use_img_input']
    
    """image_path = "/home/ubuntu/thesis/data/samples/plots/agriculture_0.jpeg"
    prompt = "Please describe this time series about the yearly Aggregated input index (2015=100) in Senegal, in the context of agriculture. Starting from 2008 and ending in 2013. Answer in a single concise paragraph, without formatting."
    
    print(get_vlm_response(model_name="gemini-2.0-flash", prompt=prompt, image_path=image_path))"""
    
    
    model_name = config['eval']['evaluated_model']
    
    ts_folder_path = "/home/ubuntu/thesis/data/samples/len 300/time series"
    metadata_folder_path = "/home/ubuntu/thesis/data/samples/len 300/metadata"
    image_folder_path = "/home/ubuntu/thesis/data/samples/len 300/plots"
    save_folder_path = f"/home/ubuntu/thesis/data/samples/len 300//generated captions/{model_name}{"" if use_img_input else "_text"}"
    
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
        
        with open(ts_path, "r") as ts_file:
            ts = [float(line.strip()) for line in ts_file]
            
        with open(metadata_path, "r") as metadata_file:
            metadata = json.load(metadata_file)
        
        prompt = generate_prompt_for_baseline(dataset_name=dataset_name, metadata=metadata, ts=ts)
        if use_img_input:
            prompt = prompt + "\nI have attached a line plot of the time series to support you."
            if "claude" in model_name:
                generated_caption = get_claude_image_response(image_path, prompt)
            else:
                generated_caption = get_vlm_response(model_name=model_name, prompt=prompt, image_path=image_path)
            
        else:
            if "claude" in model_name: 
                generated_caption = get_claude_response(prompt)
            else:
                if model_name == "gemini-2.0-flash": 
                    model = "Google Gemini-2.0-Flash"
            
                generated_caption = get_response(prompt=prompt, model=model)
        
        #print(generated_caption)
        save_file(data=generated_caption, filepath=save_folder_path+"/"+filename[:-4]+"txt")
        
        if i % 50 == 0 and i != 0:
            print(f"\n{i}/{len(filenames)} Done.")
        
        
    
if __name__ == "__main__":
    main()
    

