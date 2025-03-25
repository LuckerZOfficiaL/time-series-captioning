import requests 
import random 
import numpy as np 
import pandas as pd
import openai
import json
import matplotlib.pyplot as plt
import boto3
from concurrent.futures import ThreadPoolExecutor
from google import genai
from google.genai import types
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch
import os
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import re
import yaml
import spacy
import nltk
from nltk.corpus import wordnet
from llm_axe.agents import OnlineAgent
from llm_axe.models import OllamaChat
from llm_axe import OnlineAgent, OllamaChat
import re
from typing import Optional


def load_config(filepath="/home/ubuntu/thesis/source/configs/config.yaml"):
    with open(filepath, "r") as file:
        config = yaml.safe_load(file)
    return config

def get_response(prompt,
                 system_prompt="You are a helpful assistant and you have to generate text on my request.",
                 model="OpenAI GPT-4o",  # "Gemini-1.5-Pro"
                 temperature=0.45,  # Controls randomness (0 = deterministic, 1 = max randomness)
                 top_p=.95,  # Nucleus sampling (0.0 to 1.0, lower = more focused sampling)
                 top_k=40,  # Filters to the top-k highest probability tokens (if supported)
                 max_tokens=450,
                 online=False  # Maximum number of tokens in response
                 ):

    # Check if prompt is a list or a single string
    is_list = isinstance(prompt, list)
    prompts = prompt if is_list else [prompt]  # Ensure we always work with a list

    responses = []

    def process_prompt(p):
        if model == "OpenAI GPT-4o":
            # Read OpenAI API key
            with open("/home/ubuntu/thesis/.credentials/openai", "r") as file:
                openai_api_key = file.read().strip()

            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {openai_api_key}"
            }

            data = {
                "model": "gpt-4o",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": p}
                ],
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens
            }

            response = requests.post(url, headers=headers, json=data)

            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            else:
                print("Error:", response.status_code, response.text)
                return None

        elif model == "Anthropic Claude-3.5":
            bedrock = boto3.client(service_name="bedrock-runtime", region_name="us-east-2")

            input_payload = {
                "modelId": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
                "contentType": "application/json",
                "accept": "*/*",
                "body": json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "messages": [{"role": "user", "content": p}],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p
                })
            }

            response = bedrock.invoke_model(
                body=input_payload["body"],
                modelId=input_payload["modelId"],
                accept=input_payload["accept"],
                contentType=input_payload["contentType"],
            )

            response_body = json.loads(response["body"].read().decode("utf-8"))
            return response_body['content'][0]['text']

        elif model == "Google Gemini-2.0-Flash" or model == "Online Gemini-2.0-Flash":
          with open("/home/ubuntu/thesis/.credentials/google", "r") as file:
              google_api_key = file.read().strip()
          client = genai.Client(api_key=google_api_key)

          online = False
          if "online" in model.lower():
            onlie = True

          tools = []
          if online:
            google_search_tool = Tool(
              google_search = GoogleSearch()
            )
            tools = [google_search_tool]
     
          response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=p,
            config=types.GenerateContentConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
                tools=tools,
                response_modalities=["TEXT"],
            )
          )
          text_response = response.text
          #web_metadata = response.candidates[0].grounding_metadata.search_entry_point.rendered_content # To get grounding metadata as web content.
          return text_response

        elif "Ollama" in model:
          if "llama3.3" in model: model_name = "llama3.3"
          elif "gemma3" in model: model_name = "gemma3:27b"
          elif "mixtral 8x7b" in model: model_name = "mixtral:8x7b"
          elif "mixtral 8x22b" in model: model_name = "mixtral:8x22b"
          elif "qwen2.5" in model: model_name = "myaniu/qwen2.5-1m"
          elif "nemotron" in model: model_name = "nemotron"
          llm = OllamaChat(model=model_name)
          online_agent = OnlineAgent(llm)

          resp = online_agent.search(p)
          resp = resp.lstrip()
          if resp.startswith("Based on information from the internet, "):
            resp = resp[len("Based on information from the internet, "):]
          return resp

        else:  # the model is one of the self-hosted
            with open("/home/ubuntu/thesis/.credentials/openai", "r") as file:
                API_KEY = file.read().strip()

            API_ENDPOINT = "https://backend.zzhou.info/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            }

            data = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": p}
                ],
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens
            }

            if top_k is not None:
                data["top_k"] = top_k

            response = requests.post(API_ENDPOINT, headers=headers, json=data)

            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                print("Error:", response.status_code, response.text)
                return None

    with ThreadPoolExecutor() as executor:
        responses = list(executor.map(process_prompt, prompts))

    # Return a single response if the input was a single string, otherwise return a list
    return responses if is_list else responses[0]

# This is the old code that feeds prompts one by one. Deprecated!
def get_response_iterative(prompt, 
                 system_prompt="You are a helpful assistant and you have to generate text on my request.",
                 model="GPT-4o",  # "Gemini-1.5-Pro"
                 temperature=0.75,  # Controls randomness (0 = deterministic, 1 = max randomness)
                 top_p=.95,  # Nucleus sampling (0.0 to 1.0, lower = more focused sampling)
                 top_k=40,  # Filters to the top-k highest probability tokens (if supported)
                 max_tokens=450,  # Maximum number of tokens in response
                 ):

    # Check if prompt is a list or a single string
    is_list = isinstance(prompt, list)
    prompts = prompt if is_list else [prompt]  # Ensure we always work with a list

    responses = []

    if model == "OpenAI GPT-4o":
        # Read OpenAI API key
        with open("/home/ubuntu/thesis/.credentials/openai", "r") as file:
            openai_api_key = file.read().strip()

        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai_api_key}"
        }

        for p in prompts:
            data = {
                "model": "gpt-4o",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": p}
                ],
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens
            }

            response = requests.post(url, headers=headers, json=data)

            if response.status_code == 200:
                responses.append(response.json()['choices'][0]['message']['content'])
            else:
                print("Error:", response.status_code, response.text)
                responses.append(None)

    elif model == "Anthropic Claude-3.5":
        bedrock = boto3.client(service_name="bedrock-runtime", region_name="us-east-2")

        for p in prompts:
            input_payload = {
                "modelId": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
                "contentType": "application/json",
                "accept": "*/*",
                "body": json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "messages": [{"role": "user", "content": p}],
                    "max_tokens": 450,
                    "temperature": temperature,
                    "top_p": top_p
                })
            }

            response = bedrock.invoke_model(
                body=input_payload["body"],
                modelId=input_payload["modelId"],
                accept=input_payload["accept"],
                contentType=input_payload["contentType"],
            )

            response_body = json.loads(response["body"].read().decode("utf-8"))
            responses.append(response_body['content'][0]['text'])

    else:  # Use the self-hosted model
        with open("/home/ubuntu/thesis/.credentials/openai", "r") as file:
            API_KEY = file.read().strip()

        API_ENDPOINT = "https://backend.zzhou.info/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }

        for p in prompts:
            data = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": p}
                ],
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens
            }

            if top_k is not None:
                data["top_k"] = top_k

            response = requests.post(API_ENDPOINT, headers=headers, json=data)

            if response.status_code == 200:
                responses.append(response.json()["choices"][0]["message"]["content"])
            else:
                print("Error:", response.status_code, response.text)
                responses.append(None)

    # Return a single response if the input was a single string, otherwise return a list
    return responses if is_list else responses[0]

def rank_responses(responses_list: list, model="GPT-4o") -> list: # takes a list of texts, returns a ranking of the indices
  unified_responses = ""
  for i in range(len(responses_list)):
    unified_responses += str(i+1) + ". " + responses_list[i] + "\n\n"

  request = """The following are descriptions of the same time series.
                Rank them from the best to the worst, according to informativeness, factual accuracy, information redundancy, and the use of external knowledge.
                Answer only with the ranked indices directly and don't say anything more, don't copy the entire descriptions.
            """

  ranked_responses = get_response(request + unified_responses, model)
  ranked_responses = ranked_responses.split(",")
  ranked_responses = [int(x) for x in ranked_responses]
  return ranked_responses

#rank_responses(["The time series started from 5.2 and dropped to 2.9", "The time series is declining", "The time series describes the daily temperatures of Paris, dropping from 5.2 to 2.9 in 10 days."])

def get_sample(dataset_name: str, json_data, series_len = None, start_idx = None): # returns the metadata and the time series
  if dataset_name == "air quality":
    id = random.choice(list(json_data.keys()))
    #print("\nID: ", id)
    #print("\nKeys: ", json_data[id].keys())
    choices = list(json_data[id].keys())
    choices.remove("metadata")
    measure = random.choice(choices)
    #print("\nMeasure: ", measure)

    if series_len is None:
      series_len = random.randint(5, min(150, 5+int(len(json_data[id][measure])/8)))
    if start_idx is None:
      start_idx = random.randint(0, len(json_data[id][measure]) - series_len)
    #print("series len ", series_len)
    #print("start idx", start_idx),
    #print("tot series len", len(json_data[id][measure]))
    try:
      ts = json_data[id][measure][start_idx:start_idx+series_len]
      ts = [round(x, 2) for x in ts]

    except KeyError as e:
      print(e)
      print("series len ", series_len)
      print("start idx", start_idx),
      print("tot series len", len(json_data[id][measure]))
      

    metadata = json_data[id]["metadata"].copy()
    metadata_cpy = metadata.copy()

    attributes_to_keep = ['state', 'city', 'station_location','start_month','start_year','mean','standard deviation','min','max','starting time']

    for attr in metadata_cpy:
      if attr not in attributes_to_keep:
        del metadata[attr]

    metadata["measure"] = measure
    metadata["mean"] = round(metadata_cpy["mean"][measure], 2)
    metadata["standard deviation"] = round(metadata_cpy["std"][measure], 2)
    metadata["min"] = round(metadata_cpy["min"][measure], 2)
    metadata["max"] = round(metadata_cpy["max"][measure], 2)


    metadata["all-time average value until today"] = round(metadata.pop("mean"), 2)
    metadata["all-time standard deviation until today"] = round(metadata.pop("standard deviation"), 2)
    metadata["all-time minimum"] = round(metadata.pop("min"), 2)
    metadata["all-time  maximum"] = round(metadata.pop("max"), 2)
    metadata["starting time"] = metadata["starting time"][start_idx]

    metadata['average value in this time series'] = round(np.mean(ts), 2)
    metadata['standard deviation in this time series'] = round(np.std(ts), 2)
    metadata['minimum value in this time series'] = round(min(ts), 2)
    metadata['maximum value in this time series'] = round(max(ts), 2)

    metadata["sampling frequency"] = "hourly"

  elif dataset_name == "crime":
    town = random.choice(list(json_data.keys()))
    metadata = json_data[town]['metadata'].copy()
    if series_len is None:
      series_len = random.randint(5, min(150, 5+int(len(json_data[town]["data"])/8)))
    if start_idx is None:
      start_idx = random.randint(0, len(json_data[town]["data"]) - series_len)

    ts = json_data[town]['data'][start_idx:start_idx + series_len]
    ts = [round(x, 2) for x in ts]

    metadata["start date of the series"] = json_data[town]['metadata']['start date'][:-9]
    date = pd.to_datetime(metadata["start date of the series"])
    start_date = date + pd.DateOffset(days=start_idx)
    end_date = start_date + pd.DateOffset(days=series_len)
    metadata["start date of the series"] =  start_date.strftime('%Y-%m-%d')
    metadata["end date of the series"] =  end_date.strftime('%Y-%m-%d')

    metadata["sampling frequency"] = "daily"
    metadata['series length'] = series_len
    metadata["general mean in the history of this town"] = round(json_data[town]['metadata']['mean'], 2)
    metadata["general standard deviation in the history of this town"] = round(json_data[town]['metadata']['std'], 2)
    metadata["general minimum in the history of this town"] = round(json_data[town]['metadata']['min'], 2)
    metadata["general maximum in the history of this town"] = round(json_data[town]['metadata']['max'], 2)

    metadata["mean of this specific series"] = round(np.mean(ts), 2)
    metadata["standard deviation of this specific series"] = round(np.std(ts), 2)
    metadata["minimum of this specific series"] = round(min(ts), 2)
    metadata["maximum of this specific series"] = round(max(ts), 2)

    del metadata['min']
    del metadata['max']
    del metadata['mean']
    del metadata['std']
    del metadata['start date']
    del metadata['end date']

  elif dataset_name == "border crossing":
    port = random.choice(list(json_data.keys()))

    metadata = {}
    means = random.choice(list(json_data[port]['data'].keys()))

    if series_len is None:
      series_len = random.randint(5, min(150, 5+int(len(json_data[port]["data"][means])/8)))
    if start_idx is None:
      start_idx = random.randint(0, len(json_data[port]["data"][means]) - series_len)

    ts = json_data[port]['data'][means][start_idx:start_idx + series_len]


    metadata['port'] = port
    metadata['means'] = means

    metadata["state"] = json_data[port]['metadata']['state']
    metadata["border"] = json_data[port]['metadata']['border']
    metadata["sampling frequency"] = "monthly"
    metadata["start date of the series"] = json_data[port]['metadata']['start date'][:-9]
    date = pd.to_datetime(metadata["start date of the series"])
    start_date = date + pd.DateOffset(months=start_idx)
    end_date = start_date + pd.DateOffset(months=series_len)
    metadata["start date of the series"] =  start_date.strftime('%Y-%m-%d')
    metadata["end date of the series"] =  end_date.strftime('%Y-%m-%d')

    metadata["general mean in the history of this port"] = round(json_data[port]['metadata']['mean'][means], 2)
    metadata["general standard deviation in the history of this port"] = round(json_data[port]['metadata']['std'][means], 2)
    metadata["general minimum in the history of this port"] = round(json_data[port]['metadata']['min'][means], 2)
    metadata["general maximum in the history of this port"] = round(json_data[port]['metadata']['max'][means], 2)

    metadata['mean of this specific series'] = round(np.mean(ts), 2)
    metadata['standard deviation of this specific series'] = round(np.std(ts), 2)
    metadata['minimum in this specific series'] = round(min(ts), 2)
    metadata['maximum in this specific series'] = round(max(ts), 2)

  elif dataset_name == "heart rate":
    patient_id = random.choice(list(json_data.keys()))
    metadata = {}
    
    if series_len is None:
      series_len = random.randint(5, min(150, 5+int(len(json_data[patient_id]["data"]["heart rate"])/8)))
    if start_idx is None:
      start_idx = random.randint(0, len(json_data[patient_id]["data"]["heart rate"]) - series_len)

    ts = json_data[patient_id]['data']['heart rate'][start_idx:start_idx + series_len]
    ts = [round(x, 2) for x in ts]


    metadata['general mean of this patient in this situation'] = round(json_data[patient_id]['metadata']['mean'], 2)
    metadata['general standard deviation of this patient in this situation'] = round(json_data[patient_id]['metadata']['std'], 2)
    metadata['general minimum of this patient in this situation'] = round(json_data[patient_id]['metadata']['min'], 2)
    metadata['general maximum of this patient in this situation'] = round(json_data[patient_id]['metadata']['max'], 2)
    metadata['mean of this specific series'] = round(np.mean(ts), 2)
    metadata['standard deviation of this specific series'] = round(np.std(ts), 2)
    metadata['minimum of this specific series'] = round(min(ts), 2)
    metadata['maximum of this specific series'] = round(max(ts), 2)

    if "." in patient_id:
      unpacked_id = patient_id.split(".")
    else:
      unpacked_id = [patient_id]

    if len(unpacked_id) == 1:
      category_letter = unpacked_id[0][0]
      if category_letter == "N":
        metadata['category'] = "normal person"
      elif category_letter == "M":
        metadata['category'] = "metronomic breathing practitioner"
      elif category_letter == "I":
        metadata['category'] = "elite triathlon athlete"
      elif category_letter == "Y":
        metadata['category'] = "yoga practitioner"

    elif len(unpacked_id) == 2:
      category_letter = unpacked_id[0][0]
      if category_letter == "Y":
        metadata['category'] = "yoga meditation practitioner"
      elif category_letter == "C":
        metadata['category'] = "chi meditation practitioner"

      moment = unpacked_id[1]
      if moment == "pre":
        metadata['moment'] = "before meditation"
      elif moment == "med":
        metadata['moment'] = "during meditation"

  elif dataset_name == "demography":
    #series_len = 22 # let's fix it at 22 because we only have 22 timesteps for any country
    country_ID = random.choice(list(json_data.keys()))
    attribute = random.choice([key for key in json_data[country_ID].keys() if key != "metadata"])

    if series_len is None:
      series_len = random.randint(5, len(json_data[country_ID][attribute]))
    if start_idx is None:
      start_idx = random.randint(0, len(json_data[country_ID][attribute]) - series_len)

    metadata = {}

    metadata['country'] = json_data[country_ID]['metadata']['country name']
    metadata['attribute'] = attribute
    metadata['category by income'] = json_data[country_ID]["metadata"]['By Income']
    metadata['groups'] = json_data[country_ID]["metadata"]['Other Country Groups']
    if len(metadata['groups']) == 0: del metadata['groups']
    metadata['starting year'] = json_data[country_ID]["metadata"]['start year of the series'] + start_idx
    metadata['end year'] = metadata['starting year'] + series_len - 1
    metadata['sampling frequency'] = "yearly"

    ts = json_data[country_ID][attribute][start_idx:start_idx+series_len]
    average_ts = np.mean(
        [json_data[country][attribute][start_idx:start_idx+series_len] 
        for country in json_data if country != country_ID and 
        not np.any(np.isnan(json_data[country][attribute][start_idx:start_idx+series_len]))], 
        axis=0
    )
    ts = [round(x, 2) for x in ts]
    metadata['global average time series'] = [round(x, 2) for x in average_ts]
    metadata['global standard deviation'] = round(np.std(metadata['global average time series']), 2)

    metadata['mean of this specific series'] = round(np.mean(ts), 2)
    metadata['standard deviation of this specific series'] = round(np.std(ts), 2)
    metadata['minimum of this specific series'] = round(min(ts), 2)
    metadata['maximum of this specific series'] = round(max(ts), 2)

  return metadata, ts

# the following function does not preclude that no sample is duplicated, there's a very slim chance that it occurs
def get_samples(dataset_name, json_data, n, series_len=None) -> list: # returns a list of tuples (metadata, ts) of the specified dataset
  samples = []
  if n is not None: # this fixes the number of samples
    i = 0
    while i < n:
      metadata, ts = get_sample(dataset_name, json_data, series_len=None)
      if not np.isnan(ts).any() and not any(isinstance(x, str) and x.lower() == 'nan' for x in ts):
        zero_percentage = (ts.count(0) / len(ts)) * 100
        if zero_percentage <= 10:
            samples.append((metadata, ts))
            i += 1
      
  return samples

def get_request(dataset_name, metadata, ts, external_knowledge=True):
  if dataset_name == "air quality":
    request = f"""Here is a time series about {metadata["sampling frequency"]} {metadata["measure"]} in the Indian city of {metadata['city']}: \n {ts} \n Here is the detailed metadata: \n {str(metadata)}.
          \n Describe this time series by focusing on trends and patterns. Discuss concrete numbers you see and pay attention to the dates.
          For numerical values, ensure consistency with the provided time series. If making percentage comparisons, round to the nearest whole number.Report the dates when things happened.
          Use the statistics I provided you for comparing this example to the normalcy.
          {"Use your broad knowledge of geopolitics, natural events, and economic trends to provide meaningful comparisons. Be specific and factual, avoiding broad generalizations." if external_knowledge else "Do not add any extra information beyond what is given."}
          Highlight significant spikes, dips, or patterns{" and explain possible causes based on global or regional factors." if external_knowledge else "."}
          You don't have to explicitly report the numeric values of general statistics, you just use them for reference.
          Compare the trends in this time series to global or regional norms, explaining whether they are higher, lower, or follow expected seasonal patterns.
          When making comparisons, clearly state whether differences are minor, moderate, or significant.
          Use descriptive language to create engaging, natural-sounding text.
          Avoid repetitive phrasing and overused expressions.

          Answer in a single paragraph of four sentences at most, without bullet points or any formatting.

          """
  
  elif dataset_name == "crime":
    request = f"""Here is a time series about the number of {metadata["sampling frequency"]} crimes {metadata["town"]}, Los Angeles, from {metadata["start date of the series"]} to {metadata["end date of the series"]}: \n {ts}
          \nThe all-time statistics of {metadata["town"]} until today are: \n Mean: {metadata["general mean in the history of this town"]} \n Standard Deviation: {metadata["general standard deviation in the history of this town"]} \n Minimum: {metadata["general minimum in the history of this town"]} \n Maximum: {metadata["general maximum in the history of this town"]}
          \nAnd the statistics for this specific time series are: \n Mean: {metadata["mean of this specific series"]} \n Standard Deviation: {metadata["standard deviation of this specific series"]} \n Minimum: {metadata["minimum of this specific series"]} \n Maximum: {metadata["maximum of this specific series"]}

         \nDescribe this time series by focusing on trends and patterns. Discuss concrete   numbers you see and pay attention to the dates.
          For numerical values, ensure consistency with the provided time series. If making percentage comparisons, round to the nearest whole number.Report the dates when things happened.
          Use the statistics I provided you for comparing this example to the normalcy.
          {"Use your broad knowledge of geopolitics, natural events, and economic trends to provide meaningful comparisons. Be specific and factual, avoiding broad generalizations." if external_knowledge else "Do not add any extra information beyond what is given."}
          Highlight significant spikes, dips, or patterns{" and explain possible causes based on global or regional factors." if external_knowledge else "."}
          You don't have to explicitly report the numeric values of general statistics, you just use them for reference.
          Compare the trends in this time series to global or regional norms, explaining whether they are higher, lower, or follow expected seasonal patterns.
          When making comparisons, clearly state whether differences are minor, moderate, or significant.
          Use descriptive language to create engaging, natural-sounding text.
          Avoid repetitive phrasing and overused expressions.

          Answer in a single paragraph of four sentences at most, without bullet points or any formatting.

          """

  elif dataset_name == "border crossing":
    request = f"""Here is a time series about the number of {metadata['sampling frequency']} {metadata['means']} crossing the port of {metadata['port']} at the {metadata["border"]} border, starting from {metadata["start date of the series"]}: \n {ts}
          \nThe all-time statistics until today of {metadata['means']} crossing {metadata['port']} are: \n Mean: {metadata["general mean in the history of this port"]} \n Standard Deviation: {metadata["general standard deviation in the history of this port"]} \n Minimum: {metadata["general minimum in the history of this port"]} \n Maximum: {metadata["general maximum in the history of this port"]}
          Note that these all-time statistics are computed from then all the way until today. These are not historical, these are all-time.
          \nThe statistics for this specific time series are: \n Mean: {metadata['mean of this specific series']} \n Standard Deviation: {metadata['standard deviation of this specific series']} \n Minimum: {metadata['minimum in this specific series']} \n Maximum: {metadata['maximum in this specific series']}

           \n Describe this time series by focusing on trends and patterns. Discuss concrete numbers you see and pay attention to the dates.
          For numerical values, ensure consistency with the provided time series. If making percentage comparisons, round to the nearest whole number.Report the dates when things happened.
          Use the statistics I provided you for comparing this example to the normalcy.
         {"Use your broad knowledge of geopolitics, natural events, and economic trends to provide meaningful comparisons. Be specific and factual, avoiding broad generalizations." if external_knowledge else "Do not add any extra information beyond what is given."}
          Highlight significant spikes, dips, or patterns{" and explain possible causes based on global or regional factors." if external_knowledge else "."}
          You don't have to explicitly report the numeric values of general statistics, you just use them for reference.
          Compare the trends in this time series to global or regional norms, explaining whether they are higher, lower, or follow expected seasonal patterns.
          When making comparisons, clearly state whether differences are minor, moderate, or significant.
          Use descriptive language to create engaging, natural-sounding text.
          Avoid repetitive phrasing and overused expressions.

          Answer in a single paragraph of four sentences at most, without bullet points or any formatting.
          """

  elif dataset_name == "heart rate":
    request = f"""Here is a time series about the heart rate of a {metadata["category"]}{' ' + metadata["moment"] if "moment" in metadata else ''}, it's measured as instantaneous heart rates across measurements. Here it is: \n {ts}
          \nThe general statistics of this person{' ' + metadata["moment"] if "moment" in metadata else ''} are: \n Mean: {metadata['general mean of this patient in this situation']} \n Standard Deviation: {metadata['general standard deviation of this patient in this situation']} \n Minimum: {metadata['general minimum of this patient in this situation']} \n Maximum: {metadata['general maximum of this patient in this situation']}
          \nThe statistics for this specific time series are: \n Mean: {metadata['mean of this specific series']} \n Standard Deviation: {metadata['standard deviation of this specific series']} \n Minimum: {metadata['minimum of this specific series']} \n Maximum: {metadata['maximum of this specific series']}

          \n Describe this time series by focusing on trends and patterns. Discuss concrete numbers you see and pay attention to the dates.
          For numerical values, ensure consistency with the provided time series. If making percentage comparisons, round to the nearest whole number.Report the dates when things happened.
          Use the statistics I provided you for comparing this example to the normalcy.
          {"Use your broad knowledge of geopolitics, natural events, and economic trends to provide meaningful comparisons. Be specific and factual, avoiding broad generalizations." if external_knowledge else "Do not add any extra information beyond what is given."}
          Highlight significant spikes, dips, or patterns{" and explain possible causes based on global or regional factors." if external_knowledge else "."}
          You don't have to explicitly report the numeric values of general statistics, you just use them for reference.
          Compare the trends in this time series to global or regional norms, explaining whether they are higher, lower, or follow expected seasonal patterns.
          When making comparisons, clearly state whether differences are minor, moderate, or significant.
          Use descriptive language to create engaging, natural-sounding text.
          Avoid repetitive phrasing and overused expressions.

          Answer in a single paragraph of four sentences at most, without bullet points or any formatting.
          """

  elif dataset_name == "demography":
    request = f"""I will give you a time series about the {metadata['sampling frequency']} {metadata['attribute']} of {metadata['country']} from {metadata['starting year']} to {metadata['end year']}, it's measured as number per 1000 people.
          {metadata['country']} is categorized as a country with these attributes: {metadata['category by income']}.
           Here is the time series: \n {ts}
          \nHere are the statistics for this specific time series for {metadata['country']}: \n Mean: {metadata['mean of this specific series']} \n Standard Deviation: {metadata['standard deviation of this specific series']} \n Minimum: {metadata['minimum of this specific series']} \n Maximum: {metadata['maximum of this specific series']}
          \nHere is the global average time series for {metadata['attribute']} across all countries in the same period: \n {metadata['global average time series']}, whose standard deviation is {metadata['global standard deviation']}

          \n Describe this time series by focusing on trends and patterns. Discuss concrete numbers you see and pay attention to the dates.
          For numerical values, ensure consistency with the provided time series. If making percentage comparisons, round to the nearest whole number.Report the dates when things happened.
          Use the statistics I provided you for comparing this example to the normalcy.
          {"Use your broad knowledge of geopolitics, natural events, and economic trends to provide meaningful comparisons. Be specific and factual, avoiding broad generalizations." if external_knowledge else "Do not add any extra information beyond what is given."}
          Highlight significant spikes, dips, or patterns{" and explain possible causes based on global or regional factors." if external_knowledge else "."}
          You don't have to explicitly report the numeric values of general statistics, you just use them for reference.
          Compare the trends in this time series to global or regional norms, explaining whether they are higher, lower, or follow expected seasonal patterns.
          When making comparisons, clearly state whether differences are minor, moderate, or significant.
          Use descriptive language to create engaging, natural-sounding text.
          Avoid repetitive phrasing and overused expressions.

          Answer in a single paragraph of four sentences at most, without bullet points or any formatting.
          """
  return request

def augment_request(request, n=3, model="GPT-4o"): # rephrases the request prompt n times and returns the augmentations in a list
  augmentation_request = f"""
          Your task is to rephrase the given prompt while preserving all its original information, intent, meta-data, and length.
          - Ensure that the meaning remains unchanged, including instructions related to numerical accuracy, world knowledge, and comparison guidelines.
          - Generate {n} distinct variations, each with a different writing style you can pick from this list:
            1. Formal (precise and professional)
            2. Journalistic (engaging and informative)
            3. Conversational (natural and friendly)
            4. Technical (structured and rigorous)
            5. Creative (slightly varied sentence structure, but factual)

          Here is the original prompt.
          ----------------  \n\n
          {request}
          ----------------  \n\n
          Note that you don't have to answer to the original prompt but just to rephrase it in different ways and write down a singlple concise paragraph, maintain the numeric time series in the prompt.
          Separate each variant with a line without specifying the style. Start with your answer directly without saying anything else.
  """


  variants_response = get_response(augmentation_request, model=model,
                          temperature = 0.7,
                          top_p = 0.85,
                          )


  prompt_variants = variants_response.split("\n\n")
  for variant in prompt_variants:
    if len(variant) < 20: # remove artifacts that are not prompts
      prompt_variants.remove(variant)
  for i in range(len(prompt_variants)): # this request is often ignored in the augmented prompts, so let's add it back
      prompt_variants[i] += "\nAnswer in a single paragraph of four sentences at most, without bullet points or any formatting."

  return prompt_variants

def get_captions(prompt: str, model_list):
  captions = []
  for model in model_list:
    caption.append(get_response(prompt, model=model,
                          temperature = 0.7,
                          top_p = 0.85,
                  ))
  return captions

def save_file(data, filepath: str, mode= "w"):
    """
    Saves data to a file, supporting strings, lists, dictionaries, and tensors.

    Args:
        data: The data to save.
        filepath (str): The path to the file.
    """
    if isinstance(data, str):
        #print(f"Data type is string for {filepath}.")
        with open(filepath, mode) as file:
            file.write(data)
    elif isinstance(data, list):
        #print(f"Data type is list for {filepath}.")
        with open(filepath, mode) as file:
            for item in data:
                file.write(str(item) + '\n'+'_'*80+"\n")
    elif isinstance(data, dict):
        #print(f"Data type is dictionary for {filepath}.")
        with open(filepath, mode) as file:
            json.dump(data, file, indent=4, sort_keys=True)
    elif isinstance(data, torch.Tensor):
        #print(f"Data type is tensor for {filepath}.")
        torch.save(data, filepath)
    else:
        raise ValueError("Unsupported data type")

def add_facts_to_caption(caption, model="OpenAI GPT-4o", temperature=0.3, ask_urls=False):
    prompt = f"""
    Here is a time series description. Carefully analyze it:  
    \n
    {caption}  
    \n
    The description may include vague references to scientific facts, economic, or geopolitical events.  
    1. Identify any **unclear or speculative** statements.  
    2. **Replace** them with **concrete facts** by referring to your scientific knowledge and historical events from that period. Rely only on trusted sources, not fake news.
    3. For each fact added, **mention its source, historical reference, or well-documented event**.  
    {"4. If possible, provide URLs to support your statements. If not, ignore this request without commenting." if ask_urls else ""}
    
    **Rules:**  
    - Do NOT modify the original structure of the description beyond factual refinements.  
    - Ensure the information you add is correct and not fake.
    - Return ONLY the refined caption in one paragraph, do not introduce your refinement but write your refinement directly.  
    """
    
    response = get_response(prompt=prompt, model=model,
                            temperature=temperature,  # Lower temp for reliability
                            top_p=0.85)
    return response

def change_linguistic_style(caption, style="casual", model="OpenAI GPT-4o"):
    prompt = f"""
    Here is a time series description. Carefully analyze it:  
    \n
    {caption}  
    \n
    Rewrite the description using a **{style}** linguistic style while **preserving all information, numbers, and factual details**.  
    - Do **not** remove, add, or alter the meaning of the content.  
    - Adapt only the **tone, phrasing, and word choice** to match the requested style.  
    - Keep it fluent and natural.  
      
    **Return only the rewritten description. Do not include explanations or formatting.**  
    """
    
    response = get_response(prompt=prompt, model=model,
                            temperature=0.7,  # Balanced randomness for stylistic variety
                            top_p=0.9)  # Slightly more diverse phrasing
    return response

def enrich_language(caption, model="OpenAI GPT-4o"):
  prompt = f"""
  Here is a time series description, read it carefully. 
  \n
  {caption} 
  \n
  Rewrite the above description using richer and more diverse language. Avoid repetitions and redundant sentences. Answer with the refined description directly, without saying anything more.
  """
  response = get_response(prompt=prompt, model=model,
                          temperature = 0.75,
                          top_p = 0.85,

            )
  return response

def factual_checking(caption, model="Google Gemini-2.0-Flash"):
    prompt = f"""
    Here is a time series description. Carefully analyze it:  
    \n
    {caption}  
    \n
    The description may contain **inaccurate or misleading facts** about scientific, economic, or geopolitical events from that period. 
    
    Your task is to:  
    1. **Verify all claims or historical references** based on your knowledge.  
    2. **Identify incorrect or unsubstantiated facts** and replace them with accurate ones.  
    3. **Preserve the original writing style and structure**, modifying only incorrect statements.  
    4. **If a fact is unverifiable, state that it is uncertain rather than making assumptions**.  
    5. You can assume that the **facts with numbers are always accurate and verified**, so do not discard them.
    
    **Return only the modified description. Do not add explanations or formatting.**  
    """
    
    response = get_response(prompt=prompt, model=model,
                            temperature=0.25,  # Lower temp for more factual accuracy
                            top_p=0.85)
    return response

def generate_line_plot(ts, xlabel, ylabel, title, savepath, height=None, width=None): 
  figsize = (width, height) if width is not None and height is not None else None
  plt.figure(figsize=figsize)

  plt.plot(ts, marker='o', linestyle='-')  # Plot the time series
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.title(title)
  plt.grid(True)

  plt.savefig(savepath, bbox_inches='tight')  # Save the plot
  plt.close()
  
def extract_facts(caption, model="Google Gemini-2.0-Flash", return_list=False, extract_sentences=False):
  if extract_sentences:
    prompt = f"""
    You are an expert at extracting and decontextualizing factual statements from time series descriptions.

    Here is a time series description containing historical events, scientific facts, or geopolitical trends:

    {caption}

    Your task:

    1.  Identify all factual sentences within the description.
    2.  Rewrite each factual sentence to be self-contained and understandable without any surrounding context.
    3.  Ensure each rewritten sentence is verifiable independently.
    4.  Avoid referencing the time series itself, as it will not be available during fact verification.

    Formatting:

    -   Each decontextualized sentence should be on a new line.
    -   Leave an empty line between each rewritten sentence.
    -   Do not include any introductory or concluding text.

    Return only the decontextualized factual sentences, without any explanations, extra text, or formatting.
    """



  else:
    prompt = f"""
    Here is a time series description containing **historical events, scientific facts, or geopolitical trends**:  
    \n
    {caption}  
    \n
    **Your task:**  
    1. **Identify all explicit or implied facts** related to history, science, or geopolitics.  
    2. **Rewrite each fact as a self-contained statement** that can be verified independently.  
    3. **Include time information and locations** if they are relevant to the facts.  
    4. **Do NOT assume missing details**, extract only what is explicitly stated.  
      
    **Formatting:**  
    - Each fact should be on a **new line** with an empty line between facts.  
    - Write each fact as a **concise, complete sentence with full context**.  
    - Avoiding mentioning the time series but write verifiable sentences, because the time series is unavailable during fact verification.
    
    **Return only the extracted facts, without explanations, extra text, or formatting.**  
    """
    
  response = get_response(prompt=prompt, model=model,
                            temperature=0.15,
                            top_p=0.85)  

  if return_list:
      extracted_facts = response.split('\n')
      extracted_facts = [fact for fact in extracted_facts if fact != ""]
      return extracted_facts

  return response

def remove_non_checkable_facts(sentences, model="Google Gemini-2.0-Flash", return_list=True):
  sentence_str = '\n'.join(sentences)
  prompt = f"""
  You are an expert fact-checker tasked with identifying and removing non-verifiable statements.

  Here are some sentences:

  {sentence_str}

  Your task:

  1.  Analyze each sentence for verifiability. A sentence is considered non-verifiable if:
      * It expresses a subjective opinion or belief.
      * It cannot be verified using publicly available online resources.
      * It relies on context that is not provided.
  2.  Remove all non-verifiable sentences.
  3.  Return only the remaining verifiable sentences.

  Formatting:

  -   Each verifiable sentence should be on a new line.
  -   Leave an empty line between each verifiable sentence.
  -   Do not include any introductory or concluding text.

  Return only the extracted verifiable sentences, without any explanations, extra text, or formatting.
  """
      
  response = get_response(prompt=prompt, model=model,
                              temperature=0.15,
                              top_p=0.85)  

  if return_list:
      extracted_facts = response.split('\n')
      extracted_facts = [fact for fact in extracted_facts if fact != ""]
      return extracted_facts

  return response

def filter_facts(caption, model="Google Gemini-2.0-Flash"):
    prompt = f"""
    Here is a list of statements that may contain **real, false, or unverifiable** facts:  
    \n
    {caption}  
    \n
    **Your task:**  
    1. **Check each statement carefully** and determine if it is:  
       - **Real:** Can be verified through reputable sources.  
       - **False:** Contradicts known facts or evidence.  
       - **Unverifiable:** Too vague, subjective, or lacking enough details to check.  
    2. **Remove any statement that is false or unverifiable.**  
    3. **Keep only the real, verifiable statements.**  
      
    **Formatting:**  
    - List each **remaining fact on a new line**, separated by an empty line.  
    - Do **not** include explanations, labels, or extra text.  
      
    **Return only the filtered facts without additional text or explanation.**  
    """
    
    response = get_response(prompt=prompt, model=model,
                            temperature=0.15,  # Ensures minimal randomness for accuracy
                            top_p=0.85)  
    return response

def unify_facts(folder):
    """
    Reads all fact files from a folder (including nested subfolders), 
    extracts facts (one per line), and returns a list of all facts.
    """
    all_facts = []

    # Walk through all subdirectories and files
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith(".txt"):  # Process only text files
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    facts = [line.strip() for line in f if line.strip()]  # Remove empty lines
                    all_facts.extend(facts)
    return all_facts

def embed_sentences(sentence_list, model):
    """
    Embeds a list of sentences using a pretrained Sentence Transformer model.

    Args:
        sentences (list of str): The list of sentences to embed.
        model: The Sentence Transformer model to use.

    Returns:
        torch.Tensor: A tensor of shape [N, embedding_size] containing the sentence embeddings.
    """
    embeddings = model.encode(sentence_list, convert_to_tensor=True)
    return embeddings

def save_embeddings_pca(sentence_list, model, save_path):
    """
    Embeds sentences, performs PCA to reduce dimensionality to 2D, and visualizes them.

    Args:
        sentences (list of str): The list of sentences to embed.
        model_name (str): The name of the Sentence Transformer model to use.
    """
    # 1. Embed Sentences
    embeddings = model.encode(sentence_list)  # No need for tensor here, PCA works with numpy

    # 2. Perform PCA
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)

    # 3. Visualize
    plt.figure(figsize=(10, 8))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1])

    # Add labels (optional)
    for i, sentence in enumerate(sentence_list):
        plt.annotate(str(i), (reduced_embeddings[i, 0], reduced_embeddings[i, 1]))  # Label with sentence index

    plt.title("Sentence Embeddings in 2D (PCA)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def augment_prompt_with_rag(prompt: str, all_facts_list: list, all_facts_emb: torch.Tensor, embedding_model, retrieve_k=5) -> str:
    """
    Given a prompt, embed the prompt with embedding model, then find the indices of the top k most similar embeddings from all_facts_emb,
    use these indices to get the actual sentences from all_facts_list, then append these facts row by row to the prompt, resulting in the augmented
    prompt to return.
    """
    prompt_embedding = embedding_model.encode([prompt], convert_to_tensor=True).cpu()
    similarity_scores = cosine_similarity(prompt_embedding, all_facts_emb.cpu())[0]

    similarity_tuples = list(enumerate(similarity_scores)) # tuples of (fact_index, similarity_score)
    similarity_tuples.sort(key=lambda x: x[1], reverse=True) # sort by similarity score in descending order

    top_k_indices = [index for index, _ in similarity_tuples[:min(retrieve_k, len(similarity_tuples))]]
    top_k_facts = [all_facts_list[index] for index in top_k_indices]

    augmented_prompt = f"{prompt}\n\nHere are some OPTIONAL facts to consider, if they are helpful:\n"
    for fact in top_k_facts:
        augmented_prompt += "- " + fact + "\n"
    return augmented_prompt

def remove_common_sense(facts_list, out_path, model="Google Gemini-2.0-Flash", batch_size=8):
    batch_size = 8  # How many facts to feed in the prompt each time

    base_prompt = """
    Categorize and filter the following list of facts. 

    Classify each fact as one of the following:

    - Time-Specific: Relates to a particular point in time.
    - Location-Specific: Relates to a specific place.
    - Time and Location-Specific: Combines both time and location.
    - Common Sense/General Knowledge: Obvious, widely known information.

    Return only the facts that are NOT classified as "Common Sense/General Knowledge". 
    Output the remaining facts exactly as they appear in the input, each on a new line, with no additional explanation or formatting.

    Facts:
    """

    new_facts_list = []
    for i in range(0, len(facts_list), batch_size):  # Improved loop
        facts_batch = facts_list[i:i + batch_size]
        facts_prompt = "\n".join(facts_batch)
        prompt = base_prompt + facts_prompt
        response = get_response(prompt, model=model)[:-2] # [:-2] removes the empty line
        new_facts_list.extend(response.split("\n"))

    return new_facts_list

def extract_years(text, min_year=1900, max_year=2025): # takes a string and returns all the detected years. Years are 4 digits.
  text = str(text) # for safety, in case text is another data type
  years = re.findall(r'\b\d{4}\b', text)
  years = [int(year) for year in years if int(year) >= min_year and int(year) <= max_year] # remove non-year numbers and convert to int
  return  years

def split_facts_by_time(facts_list, bin_years=10): # reads through fact_list and categorizes the facts by their time period, storing all in one json file
  min_year = 9999
  max_year = -9999

  for fact in facts_list: # iterate to get the min and max years appeared in the facts
    years = extract_years(fact)  # Assuming there is a function to extract the year from the fact
    if years != []:
      min_year = min(min(years), min_year)
      max_year = max(max(years), max_year)
  
  print("\n\nMin and Max years:", min_year, max_year)

  time_periods = {} # a dictionary where keys=start of time period and values=facts in that period.
  start_year = min_year - (min_year % bin_years)
  while start_year <= max_year:
    time_periods[start_year] = []
    start_year += bin_years
  time_periods[0] = [] # add this key to store facts without year information

  for fact in facts_list:
    years = extract_years(fact)
    if len(years) == 0 : # use this key if there's no year information in the fact
      time_periods[0].append(fact)
    else:
      for year in years:
        for start_year in time_periods.keys():
          if start_year != 0:
            if int(year) >= start_year and int(year) <= start_year + bin_years: # if it's within that bin period
              time_periods[start_year].append(fact)

  for start_year in time_periods:
        time_periods[start_year] = list(set(time_periods[start_year])) # remove duplicates, because if both start and end years are in the same period, the fact gets added twice

  return time_periods 

def get_relevant_facts(start_year, end_year, bin_years=10): # only get facts that are temporally relevant
  folder_path = f"/home/ubuntu/thesis/data/fact bank/by period/{bin_years}"
  relevant_facts = []
  for root, dirs, files in os.walk(folder_path):
    for dir_name in dirs:
      if end_year is not None and start_year is not None:
        if int(dir_name)+bin_years >= start_year and int(dir_name) <= end_year+bin_years:
          subfolder_path = os.path.join(root, dir_name)
          fact_list_path = os.path.join(subfolder_path, "facts_list.txt")
          with open(fact_list_path, "r") as file:
            facts = file.read().splitlines()
            relevant_facts.extend(facts)

      elif start_year is None and end_year is not None:# start_year is unavailable) but end_year is available 
        if int(dir_name)+bin_years <= end_year: # consider all facts until end_year
          subfolder_path = os.path.join(root, dir_name)
          fact_list_path = os.path.join(subfolder_path, "facts_list.txt")
          with open(fact_list_path, "r") as file:
            facts = file.read().splitlines()
            relevant_facts.extend(facts)
      elif start_year is not None and end_year is None:
        if int(dir_name)+bin_years >= start_year: # consider all facts from the start_year on
          subfolder_path = os.path.join(root, dir_name)
          fact_list_path = os.path.join(subfolder_path, "facts_list.txt")
          with open(fact_list_path, "r") as file:
            facts = file.read().splitlines()
            relevant_facts.extend(facts)
      # else: no need to specify else because relevant_facts = [] and that is returned by the function
  return relevant_facts
 
def delete_files(target="samples"): #removes all files in the folder and its subfolders, preserving folders
  if target == "samples":
    root_path = "/home/ubuntu/thesis/data/samples"
  elif target == "fact bank":
    root_path = "/home/ubuntu/thesis/data/fact bank"
  for root, dirs, files in os.walk(root_path):
    for file in files:
      if file.endswith(".txt") or file.endswith(".json") or file.endswith(".jpeg") or file.endswith("pth"):
        file_path = os.path.join(root, file)
        os.remove(file_path)
  print(f"\nAll files are deleted in {target}")

def get_most_general_adjective(adjectives):
    """
    Given a list of adjectives in English, returns the most general one based on the number of WordNet synsets.

    Args:
        adjectives (list of str): A list of adjectives.

    Returns:
        str or None: The most general adjective, or None if the list is empty or no suitable adjective is found.
    """

    if not adjectives:
        return None
    if type(adjectives) == str:
      adjectives = [adjectives]
    
    best_word = ""
    max_synset = 0
    for adj in adjectives:
      syns = wordnet.synsets(adj)
      if len(syns) > max_synset:
        max_synset = len(syns)
        best_word = adj
    return best_word
    
def mask_facts(facts, mask_token="___"):
    """
    Automatically identifies key factual elements in a sentence and replaces 
    one with a masked token for verification.
    
    :param fact: The input factual statement.
    :param mask_token: The placeholder for masked elements.
    :return: The masked fact and masked words.
    """
    if type(facts) != list: # to make the input always a list
      facts = [facts]

    nlp = spacy.load("en_core_web_sm") # Load the English NLP model
    masked_facts = []
    masked_words = []

    are_masked = [] # a list of bools indicating whether the i-th fact has been masked or not

    for fact in facts:
      doc = nlp(fact)
      candidates = []
      for token in doc:
          # Mask adjectives (like "high" or "low") that modify a noun
          if token.pos_ in ["ADJ"] and token.dep_ in ["amod", "acomp"]:
              candidates.append(token.text)

          # Mask key verbs related to factual claims
          #elif token.pos_ in ["VERB"] and token.dep_ in ["ROOT"]:
          #    candidates.append(token.text)

          #if token.ent_type_ in ["GPE", "DATE", "MONEY", "PERCENT", "QUANTITY", "ORDINAL", "CARDINAL"]:
          #    candidates.append(token.text)

      # Randomly choose one element to mask (ensures variety in checks)
      if candidates:
          #print("Candidates:", candidates)
          word_to_mask = get_most_general_adjective(candidates)
          masked_words.append(word_to_mask)
          masked_fact = fact.replace(word_to_mask, mask_token, 1)  # Replace only the first occurrence
          masked_facts.append(masked_fact)
          are_masked.append(True)
      else:
          are_masked.append(False)

    return masked_facts, masked_words, are_masked 

def are_synonyms(word1, word2, threshold=0.7):
    """
    Checks if two words are synonyms based on their semantic similarity. Lists of words are supported too.

    Args:
        word1 (str or list of str): The first word or list of words.
        word2 (str or list of str): The second word or list of words.
        threshold (float): The similarity threshold for considering two words synonyms.

    Returns:
        bool or list of bool: True if the words are synonyms, False otherwise. If lists are provided, returns a list of booleans.
    """
    try:
        nlp = spacy.load("en_core_web_md")
    except OSError:
        print("spaCy model not found. Downloading...")
        
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_md"])
        nlp = spacy.load("en_core_web_md")

    if isinstance(word1, list) and isinstance(word2, list):
        if len(word1) != len(word2):
            raise ValueError("Lists word1 and word2 must have the same length.")

        results = []
        similarities = []
        for w1, w2 in zip(word1, word2):
            token1 = nlp(w1)
            token2 = nlp(w2)

            if token1.has_vector and token2.has_vector:
                similarity = token1.similarity(token2)
                results.append(similarity >= threshold)
                similarities.append(similarity)
            else:
                results.append(False)
        return results, similarities

    else: # we just have 2 words to compare
        token1 = nlp(word1)
        token2 = nlp(word2)

        if token1.has_vector and token2.has_vector:
            similarity = token1.similarity(token2)
            return similarity >= threshold, similarity
        else:
            return False

def fill_gap(masked_sentence, model="Google Gemini-2.0-Flash"):
  prompt = f"""Here's a sentence with a masked part. Answer with the word that fills is based on your knowledge.
  \n\n
  {masked_sentence}
  \n\n
  Answer just with a single word, without any explanation or additional text.
  """
  response = get_response(prompt=prompt, model=model,
                            temperature=0.15,
                            top_p=0.85)  

  response = response.split()[-1] # pick the last word if there are many
  #response = response[:-1] # to remove \n from the answer
  return '\n'.join(facts_list)

def filter_sentences_no_non_year_numbers(sentences):
    """
    Filters a list of sentences, removing those containing non-year numbers.

    Args:
        sentences: A list of strings representing sentences.

    Returns:
        A list of strings containing only sentences without non-year numbers.
    """

    def contains_non_year_number(sentence):
        """
        Checks if a sentence contains a number that is not a year.

        Args:
            sentence: A string representing a sentence.

        Returns:
            True if the sentence contains a non-year number, False otherwise.
        """
        numbers = re.findall(r'\b\d+\b', sentence)  # Find all whole numbers

        for num_str in numbers:
            num = int(num_str)
            if not (1800 <= num <= 2025):
                return True  # Found a non-year number
        return False  # No non-year numbers found

    filtered_sentences = [
        sentence for sentence in sentences if not contains_non_year_number(sentence)
    ]
    return filtered_sentences

def correct_facts_llm(facts_list: list[str], model="Google Gemini-2.0-Flash", batch_size=5, skip_numeric=True):
    """
    Checks and corrects factual inaccuracies in a list of statements using an LLM.

    Args:
        facts_list (list[str]): A list of statements to check and correct.
        model (str): The name of the LLM to use.
        batch_size (int): The number of statements to process in each batch.

    Returns:
        list[str]: A list of corrected statements.
    """

    if not facts_list:
        return []  # Return an empty list if input is empty

    if skip_numeric: # for facts with numbers, skip the checking because numbers from metadata are always correct
      facts_list = filter_sentences_no_non_year_numbers(facts_list)
      #facts_list = [fact for fact in facts_list if not any(char.isdigit() for char in fact)]

    batched_facts = [facts_list[i:i + batch_size] for i in range(0, len(facts_list), batch_size)]
    all_corrected_facts = []

    for batch in batched_facts:
        fact_str = '\n'.join(batch)
        prompt = f"""
        You are an expert fact-checker specializing in geopolitics, society, and history.

        Here are some statements, some of which may be inaccurate or unverifiable. 
        Statements containing numbers must be preserved because they are accurate and verified already. 
        Beyond numeric statements, identify the inaccurate statements and correct them with accurate information from your knowledge. Facts that are true must be left untouched.

        Output the true and corrected statements exactly as they should appear, each on a new line, with no additional explanations. 

        Statements:
        {fact_str}
        """
        try:
            response = get_response(prompt=prompt, model=model, temperature=0.2, top_p=0.85)
            corrected_facts_list = [fact.strip() for fact in response.split("\n") if fact.strip()]
            all_corrected_facts.extend(corrected_facts_list)
        except Exception as e:
            print(f"Error processing batch: {e}")
            all_corrected_facts.extend(batch)  # Keep the original facts if there's an error

    return all_corrected_facts

def extract_and_correct_facts(caption: str, method="llm", 
                              model="Google Gemini-2.0-Flash",
                              synonym_thresh = 0.7,
                              skip_numeric=True,
                              extract_sentences=True):
  facts_list = extract_facts(caption, model=model, return_list=True, extract_sentences=extract_sentences)
  #print("\nFacts list extracted:\n", facts_list)
  facts_list = remove_non_checkable_facts(facts_list, model=model)
  #print("\nFacts list without non-checkable:\n", facts_list)

  """print("\nOriginal Facts:")
  for fact in facts_list:
    print(fact)"""

  if method == "fill in the gap":
    masked_facts, masked_words, are_masked =  mask_facts(facts_list)
    #print("\nMasked facts: ", masked_facts)
    #print("\nMasked words: ", masked_words)
    filled_words = []
    for masked_fact in masked_facts:
      filled_words.append(fill_gap(masked_fact))

    #print("\nFilled words: ", filled_words)
    are_synonyms_list, similarities = are_synonyms(filled_words, masked_words, synonym_thresh) 
    #print("\nAre synonyms: ", are_synonyms_list)

    for i in range(len(masked_facts)):
      if not are_synonyms_list[i]: # the i-th filled word is not a synonym of the original word, i.e. the original fact was false, replace the original word with the new real word
        masked_facts[i] = masked_facts[i].replace("___", filled_words[i])
      else:
        masked_facts[i] = masked_facts[i].replace("___", masked_words[i]) # the i-th filled word is a synonym of the original word, fill with the original masked word

    for i in range(len(facts_list)): #replace the filled masked facts back to the list of all facts
      if are_masked[i]: # if the i-th fact was masked, replace it with the filled one
        facts_list[i] = masked_facts.pop(0)
      
    facts_list = [fact for fact in facts_list if fact != ""]
    return facts_list

  elif method == "llm":
    corrected_facts = correct_facts_llm(facts_list, model=model, skip_numeric=skip_numeric)
    """print("\nCorrected Facts:")
    for fact in corrected_facts:
      print(fact)"""
    return corrected_facts
 
def refine_caption_with_corrected_facts(caption, 
                                        model="Google Gemini-2.0-Flash",
                                        correction_method="llm",
                                        synonym_thresh=0.7,
                                        return_corrected_facts=False,
                                        skip_numeric=True,
                                        extract_sentences=True):
    facts_list = extract_and_correct_facts(caption, 
                                          method=correction_method, 
                                          model=model, 
                                          synonym_thresh=synonym_thresh,
                                          skip_numeric=skip_numeric)
    facts_str = "\n".join(facts_list)
    #print("Corrected Facts:\n ",facts_str)
    prompt = f"""
    You are an expert editor specializing in fact-checking time series descriptions.

    Here is a time series description:
    \n\n
    {caption}
    \n\n
    This description may contain inaccurate or unsubstantiated claims related to geopolitics, history, or society. You can assume that all numeric statements in the caption are correct because they are verified.

    Your task:
    1. Identify any factual errors in the description.
    2. Correct or remove the errors, using the following additional information if helpful:
    \n
    {facts_str}
    \n
    3. Ensure the refined description is accurate and coherent.
    4. Maintain the original style and tone of the description.

    Provide the refined time series description only, without any additional explanations.
    """
    if return_corrected_facts:
      return get_response(prompt=prompt, model=model,
                      temperature=0.3,
                      top_p=0.85), facts_list
    return get_response(prompt=prompt, model=model,
                      temperature=0.3,
                      top_p=0.85)

def read_txt_to_num_list(filepath):
  with open(filepath, 'r') as file:
    lines = file.readlines()
    lines = [float(line.strip()) for line in lines]
  return lines

def read_jpeg_to_tensor(filepath):
  image = plt.imread(filepath)
  image = np.array(image, copy=True)  # Make the array writable
  tensor = torch.from_numpy(image)
  return tensor

def read_txt_to_string(filepath):
  with open(filepath, 'r') as file:
    text = file.read()
  return text

def are_semantically_equivalent(str1, str2, model="Google Gemini-2.0-Flash"):
    """
    Determines if two strings are semantically equivalent using an LLM.

    Args:
        str1 (str): The first string.
        str2 (str): The second string.
        model (str): The LLM to use.

    Returns:
        bool or str: True if semantically equivalent, False if not, or an error message.
    """
    prompt = f"""
    You are an expert in determining the semantic equivalence of two text passages.

    Given the following two pieces of text, determine if they are absolutely semantically equivalent, with no disagreement in information.

    Text 1:
    {str1}

    Text 2:
    {str2}

    Answer with 'yes' if the two texts are semantically equivalent, and 'no' if they are not.
    """
    try:
        response = get_response(prompt=prompt, model=model, temperature=0.15).lower()

        if "yes" in response:
            return True
        elif "no" in response:
            return False
        else:
            return "Unable to determine semantic equivalence."

    except Exception as e:
        return f"Error during semantic equivalence check: {e}"

def are_semantically_conflicting(str1, str2, model="Google Gemini-2.0-Flash"):
    """
    Determines if two pieces of text present conflicting information using an LLM.

    Args:
        str1 (str): The first string.
        str2 (str): The second string.
        model (str): The LLM to use.

    Returns:
        bool or str: True if the texts conflict, False if not, or an error message.
    """
    prompt = f"""
    You are an expert in detecting semantic disagreements between two text passages.

    Given the following two pieces of text, determine if they present conflicting or contradictory information. 
    They may contain different information, but your focus is on identifying any direct conflicts or contradictions.

    Text 1:
    {str1}

    Text 2:
    {str2}

    Answer with 'yes' if the two texts present conflicting information, and 'no' if they do not.
    """
    try:
        response = get_response(prompt=prompt, model=model, temperature=0.15).lower()

        if "yes" in response:
            return True
        elif "no" in response:
            return False
        else:
            return "Unable to determine semantic disagreement."

    except Exception as e:
        return f"Error during semantic disagreement check: {e}"

def is_semantically_contained(sub_str, big_str, model="Google Gemini-2.0-Flash"):
    """
    Determines if the information in sub_str is semantically contained within big_str.

    Args:
        big_str (str): The larger string that may contain the information.
        sub_str (str): The smaller string whose information is being checked for containment.
        model (str): The LLM to use.

    Returns:
        bool or str: True if sub_str is semantically contained in big_str, False if not, or an error message.
    """
    prompt = f"""
    You are an expert in determining semantic containment between two text passages.

    Given the following two pieces of text, determine if all the information in Text 1 is semantically included within Text 2. 
    Text 2 may contain additional information beyond what is in Text 1, but this is acceptable. 
    Focus solely on whether Text 2 fully encompasses the meaning of Text 1, without any contradictions.

    Text 1 (Sub-string):
    {sub_str}

    Text 2 (Big-string):
    {big_str}

    Answer with 'yes' if all the information in Text 1 is semantically contained within Text 2, and 'no' if it is not.
    """
    try:
        response = get_response(prompt=prompt, model=model, temperature=0.15).lower()

        if "yes" in response:
            return True
        elif "no" in response:
            return False
        else:
            return "Unable to determine semantic containment."

    except Exception as e:
        return f"Error during semantic containment check: {e}"

def compare_correctness(str1: str, str2: str, model: str = "Google Gemini-2.0-Flash") -> int:
    """
    Compares the correctness of two strings using an LLM.

    Args:
        str1: The first string.
        str2: The second string.
        model: The LLM to use.

    Returns:
        1 if str1 is deemed more correct, 2 if str2 is more correct, or -1 if inconclusive.
    """

    prompt = f"""
    You are an expert in determining the correctness of factual statements.

    Given the following two pieces of text, determine which text is more factually correct. If both are correct or incorrect, it's inconclusive. One text wins if the it is true and the other is false.

    Text 1:
    {str1}

    Text 2:
    {str2}

    Provide your response in the following JSON format:
    {{
      "winner": "1" or "2" or "inconclusive"
    }}

    Respond only with the JSON object.
    """
    try:
        response_text = get_response(prompt, model=model, temperature=0.15).strip()
        match = re.search(r'"winner":\s*"(\w+)"', response_text)

        if match:
            winner = match.group(1)
            if winner == "1":
                return 1
            elif winner == "2":
                return 2
            elif winner == "inconclusive":
                return -1
            else:
                return -1 #unexpected response value
        else:
            return -1 #no match found

    except Exception as e:
        print(f"Error comparing correctness: {e}")
        return -1  # Inconclusive in case of error

def check_single_fact(fact, checking_model="Google Gemini-2.0-Flash"):
   prompt = f"""
     Here is a statement, your task is to check whether it's true, falase or inconclusive.
     \n
    {fact}
    \n

    Answer with either "true", "false", or "inconclusive", without adding any more text. If the statement is not always but generally true, still consider it as true.
  """
   response = get_response(prompt, model=checking_model, temperature=0.15).lower()
   if "true" in response and ("false" not in response or response.index("true") < response.index("false")):
     return True
   elif "false" in response and ("true" not in response or response.index("false") < response.index("true")):
     return False
   else:
     return None

def check_single_fact_confidence(fact, checking_model="Google Gemini-2.0-Flash"):
  prompt = f"""Please analyze the following statement and determine its factual correctness. Provide a confidence score (on a scale of 0 to 100, where 0 is completely incorrect and 100 is completely correct) for your assessment.

  Statement: {fact}

  Provide your response in the following format:

  Factual Correctness: True/False
  Confidence Score: 0-100"""

  response = get_response(prompt, model=checking_model, temperature=0.15)
  #print("Response: ", response)
  # Extract factual correctness and confidence as two variables
  correctness = None
  confidence = None

  # Check the response for correctness and confidence
  if "true" in response.lower():
    correctness = True
  elif "false" in response.lower():
    correctness = False

  # Extract the confidence score from the response
  confidence_index = response.lower().find("confidence score:")
  if confidence_index != -1:
    confidence_start = confidence_index + len("confidence score:")
    confidence_end = response.find("]", confidence_start)
    confidence = int(response[confidence_start:confidence_end])

  return correctness, confidence

def check_whole_caption(caption, extraction_model="Google Gemini-2.0-Flash", checking_model="Google Gemini-2.0-Flash", words_to_skip=[], tolerate_inconclusive=True):
  extracted_facts = extract_facts(caption, model=extraction_model, return_list=True)
  extracted_facts = filter_sentences_no_non_year_numbers(extracted_facts)
  extracted_facts = [fact for fact in extracted_facts if not any(word in fact for word in words_to_skip)]
  #print(extracted_facts)
  is_true = True
  for fact in extracted_facts:
      try:
          outcome = check_single_fact(fact, checking_model=checking_model)
          if outcome == False:
              is_true = False
              #print("False: ", fact)
              break
          elif outcome is None:
            if tolerate_inconclusive:
              pass
              #print("Inconclusive: ", fact)
            else:
              is_true = False
              print("Inconclusive!")
              break
      except Exception as e:
          #print(f"\nGot Exception on fact:\n{fact} \n{e} ")
          is_true = False
          break      
  if not is_true:
    return False, fact                   
  return is_true, None

def check_whole_caption_confidence(caption, extraction_model="Google Gemini-2.0-Flash", checking_model="Google Gemini-2.0-Flash", words_to_skip=[], confidence_thresh=60):
  extracted_facts = extract_facts(caption, model=extraction_model, return_list=True)
  extracted_facts = filter_sentences_no_non_year_numbers(extracted_facts)
  extracted_facts = [fact for fact in extracted_facts if not any(word in fact for word in words_to_skip)]
  #print(extracted_facts)
  is_true = True
  for fact in extracted_facts:
      try:
          outcome, confidence = check_single_fact_confidence(fact, checking_model=checking_model)
          if outcome == False:
              is_true = False
              #print("False: ", fact)
              break
          elif outcome is True:
            if confidence >= confidence_thresh: # a fact is true if it's classified as true with at least some confidence
              pass
              #print("Inconclusive: ", fact)
            else:
              is_true = False
              print("Unexpected outcome!")
              break
      except Exception as e:
          #print(f"\nGot Exception on fact:\n{fact} \n{e} ")
          is_true = False
          break      
  if not is_true:
    return False, fact # return False along with the fact that is either false or problematic                   
  return True, None # return True and None because no fact was problematic

def remove_source(text):
    modified_text = re.sub(r'\s*\(Source: .*?\)', '', text)
    return modified_text

def main():
  config = load_config()

  random.seed(config['general']['random_seed'])

  #print(check_single_fact_confidence("The sun is smaller than the Earth", checking_model="Google Gemini-2.0-Flash"))

  caption = """Between 2007 and 2016, the Syrian Arab Republic experienced a noticeable decline in birth rates, dropping from 30.78 to 18.87 births per 1000 people. This trend is significantly sharper than the global average decline in birth rates for the same period, which saw a decrease from approximately 20.0 to 18.5 births per 1000 people globally (World Bank Data, 2007-2016). The steepest declines occurred after 2011, coinciding with the onset of the Syrian Civil War, a conflict that began following the Arab Spring uprisings and escalating government crackdowns on protests (BBC News, "Syria: The story of the conflict"). This likely contributed to the reduced birth rate due to the resulting humanitarian crisis, including over 5 million registered refugees by 2016 (UNHCR data) and internal displacement of over 6 million people (IDMC data). Compared to the average decline in low and middle-income countries, which experienced a more moderate decrease from around 25 to 22 births per 1000 people (World Bank Data, 2007-2016), Syria's decline was both more rapid and more pronounced, indicating the profound impact of regional instability and the specific consequences of the Syrian Civil War on demographic trends."""

  """facts = extract_facts(caption, model="Ollama", return_list=True)
  print("Original facts: ", facts)
  facts = filter_sentences_no_non_year_numbers(facts)
  print("Without numeric facts: ", facts)

  for fact in facts:
    print("\n", fact, "\n", check_single_fact(fact, checking_model="Ollama llama3.3"))"""

  #print(get_response("Is the sun bigger than the Earch?", model="Ollama llama3.3", temperature=0.2))

  #print(check_whole_caption_confidence('Football is globally the most popular sport. The global population has halved in the last decade. The Chinese population has been increasing drastically lately.', extraction_model="Google Gemini-2.0-Flash", checking_model="Ollama llama3.3", confidence_thresh=0.7))


  prompts = ["Continue this sequence: 1, 4, 9, 16",
              "Who is the president of Italy in 2016?",
              "What is the best national park in California?"]

  responses = get_response(prompts, model="Ollama mixtral")
  print(responses)

  
  """facts = ["The Canadian dollar had a relatively low exchange rate against USD in 2007.",
          "The exchange rate of Canadian dollar against USD was high in 2007."]
  masked_facts, masked_words = mask_facts(facts)
  print(masked_facts)"""

  #delete_files(target="samples")

  caption = "From 2002 to 2018, Spain's birth rate per 1,000 people displayed a noticeable decline, starting at 10.1 in 2002 and dropping to 7.9 by 2018. This trend contrasts sharply with the global average, which was 19.6 per 1,000 people in 2002 and decreased to 18.5 by 2018 (World Bank Data). The most pronounced decline in Spain occurred after 2008, coinciding with the global financial crisis triggered by the collapse of Lehman Brothers in September 2008 (Lehman Brothers Bankruptcy Filing, September 2008), which led to a severe recession in Spain, characterized by high unemployment rates, particularly among young adults (Instituto Nacional de Estadística, Spain). Despite Spain's status as a high-income country, with a GNI per capita of $25,830 in 2018 (World Bank Data), its birth rate consistently fell below the global average, reflecting broader European trends of aging populations and lower fertility rates, such as Italy's rate of 7.3 per 1,000 in 2018 (Eurostat). Italy is a low income country."

  """corrected_facts = extract_and_correct_facts(caption, method="llm")
  for fact in corrected_facts:
    print(fact)"""

  """refined_caption, corrected_facts = refine_caption_with_corrected_facts(caption, 
                            model=config['model']['refinement_model'],
                            synonym_thresh=config['nlp']['synonym_similarity_thresh'],
                            return_corrected_facts=True)

  print("\nOriginal caption: ", caption)
  print("\nRefined caption: ", refined_caption)

  print("\nCorrected facts: ")
  for fact in corrected_facts:
    print(fact)"""


if __name__ == "__main__":
  main()