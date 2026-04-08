from google import genai
from google.genai import types
import base64
import vertexai
import os

def text_generate(prompt, model_name="gemini-2.5-pro-preview-05-06", temperature=0.3):
  client = genai.Client(
      vertexai=True,
      project="ts-captioning",
      location="us-west1",
  )

  model = model_name
  contents = [
      types.Content(
          role="user",
          parts=[
              types.Part(text=prompt)
          ]
      )
  ]

  generate_content_config = types.GenerateContentConfig(
      temperature = temperature,
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

  print(response.text)



def img_generate(prompt, img_paths, model_name="gemini-2.5-pro-preview-05-06", temperature=0.3):
    if type(img_paths) != list: img_paths = [img_paths]
    
    client = genai.Client(
        vertexai=True,
        project="ts-captioning",  # Replace with your project ID if different
        location="us-central1",   # Replace with your location if different
    )

    image_parts = []
    filenames = []
    for img_path in img_paths:
        filenames.append(os.path.basename(img_path)) # Get just the filename
        with open(img_path, "rb") as img_file:
            img_bytes = img_file.read()
        image_part = types.Part.from_bytes(
            data=img_bytes,
            mime_type="image/jpeg", # Assuming JPEG, adjust if necessary
        )
        image_parts.append(image_part)

    # Construct the prompt to include filenames
    # Example: "You are given the following images: image1.jpg, image2.png. {user's original question}"
    filename_str = ", ".join(filenames)
    print("Filenames: ", filename_str)
    # You can use an f-string or .format() to insert the filenames and the original prompt
    final_prompt_text = prompt + "\n\n" + "The images are named in order: " + filename_str
    #print("Prompt: ", final_prompt_text, "\n\n")


    prompt_part = types.Part.from_text(text=final_prompt_text)

    model = model_name

    # Order: typically text prompt first, then images.
    all_parts = [prompt_part] + image_parts

    contents = [
        types.Content(
            role="user",
            parts=all_parts
        ),
    ]

    contents = [
        types.Content(
            role="user",
            parts=all_parts
        ),
    ]

    generate_content_config = types.GenerateContentConfig(
        temperature = temperature,
        top_p = 1,
        seed = 0,
        max_output_tokens = 65535, # Consider if this needs adjustment based on the number of images and expected output length
        safety_settings = [types.SafetySetting(
            category="HARM_CATEGORY_HATE_SPEECH",
            threshold="BLOCK_NONE" # Using "BLOCK_NONE" as per common SDK usage if "OFF" is not a direct enum
        ),types.SafetySetting(
            category="HARM_CATEGORY_DANGEROUS_CONTENT",
            threshold="BLOCK_NONE"
        ),types.SafetySetting(
            category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
            threshold="BLOCK_NONE"
        ),types.SafetySetting(
            category="HARM_CATEGORY_HARASSMENT",
            threshold="BLOCK_NONE"
        )],
    )

    # Ensure the model you are using supports multiple image inputs in this manner.
    # Some models might have specific ways they expect multiple images (e.g., interleaved with text).
    response = client.models.generate_content( # This is likely client.generate_content for the new SDK
                                               # or model.generate_content if client.models is a specific model instance
        model = model, # For Vertex AI, this should be the full model path, e.g., "projects/YOUR_PROJECT_ID/locations/YOUR_LOCATION/endpoints/YOUR_MODEL_ENDPOINT_ID" or "gemini-..." for foundation models
        contents = contents,
        config = generate_content_config, # Note: parameter is often generation_config
    )
    return response.text
    #print(response.text)

def main():
  print(img_generate(prompt="Describe the following image or images, specifying the image filenames.", img_paths=["/home/ubuntu/thesis/data/samples/new samples no overlap/train/plots/agriculture_0_train.jpeg",
                                                                                                          "/home/ubuntu/thesis/data/samples/new samples no overlap/train/plots/demography_0_train.jpeg"]))
  
if __name__ == "__main__":
  main()
