from ast import Assert
from attr import attrib
from click import prompt
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
#from sklearn.conftest import dataset_fetchers
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import re
import yaml
import spacy
import nltk
from nltk.translate.meteor_score import meteor_score as meteor_sc
from nltk.corpus import wordnet
from llm_axe.agents import OnlineAgent
from llm_axe.models import OllamaChat
from llm_axe import OnlineAgent, OllamaChat
import re
from typing import Optional
from collections import defaultdict
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
from rouge_score import rouge_scorer
import shutil
from collections import Counter
import spacy




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
    config = load_config()
    if config['model']['temperature'] is not None:  # force the temperature to be the value in the config file
      temperature = config['model']['temperature']

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

        elif "Gemini-2.0-Flash" in model:
          with open("/home/ubuntu/thesis/.credentials/google", "r") as file:
              google_api_key = file.read().strip()
          client = genai.Client(api_key=google_api_key)

          online = False
          if "online" in model.lower():
            online = True

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
          elif "mixtral 8x7b" in moWdel: model_name = "mixtral:8x7b"
          elif "mixtral 8x22b" in model: model_name = "mixtral:8x22b"
          elif "qwen2.5-1m:14b" in model: model_name = "myaniu/qwen2.5-1m:14b"
          elif "nemotron" in model: model_name = "nemotron"
          elif "llama3.2 uncensored" in model: model_name = "artifish/llama3.2-uncensored"
          elif "qwq" in model: model_name = "qwq"
          elif "deepseek-r1:14b" in model: model_name = "deepseek-r1:14b"
          elif "phi4" in model: model_name = "phi4"
          elif "lumimaid-v0.2:12b" in model: model_name = "leeplenty/lumimaid-v0.2:12b"
          llm = OllamaChat(model=model_name)

          online_agent = OnlineAgent(llm, temperature=temperature)

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


def get_sample(dataset_name: str, json_data, is_train, series_len = None, start_idx = None): # returns the metadata and the time series
  original_series_len = series_len
  if dataset_name == "air quality":
    id = random.choice(list(json_data.keys()))
    #print("\nID: ", id)
    #print("\nKeys: ", json_data[id].keys())
    choices = list(json_data[id].keys())
    choices.remove("metadata")
    measure = random.choice(choices)
    #print("\nMeasure: ", measure)

    tot_ts_len = len(json_data[id][measure])
    train_ts_len = int(0.8*tot_ts_len)
    
    if series_len is None:
      series_len = random.randint(5, min(150, 5+int(train_ts_len/8)))
    if start_idx is None:
      if is_train: 
        start_idx = random.randint(0, train_ts_len - series_len)
      else:
        start_idx = random.randint(train_ts_len, tot_ts_len - series_len)
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
    
    tot_ts_len = len(json_data[town]["data"])
    train_ts_len = int(0.8 * tot_ts_len)
    
    if series_len is None:
      series_len = random.randint(5, min(150, 5+int(train_ts_len/8)))
    if start_idx is None:
      if is_train:
        start_idx = random.randint(0, train_ts_len - series_len)
      else:
        start_idx = random.randint(train_ts_len, tot_ts_len - series_len)

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
    while True:
      try:
        port = random.choice(list(json_data.keys()))

        metadata = {}
        means = random.choice(list(json_data[port]['data'].keys()))
        while len(json_data[port]["data"][means]) < 20:
          means = random.choice(list(json_data[port]['data'].keys()))
        
        tot_ts_len = len(json_data[port]["data"][means])
        train_ts_len = int(0.8 * tot_ts_len)

        if series_len is None:
          series_len = random.randint(5, min(150, 5+int(train_ts_len/8)))
        if start_idx is None:
          if is_train:
            start_idx = random.randint(0, train_ts_len - series_len)
          else:
            start_idx = random.randint(train_ts_len, tot_ts_len - series_len)

        ts = json_data[port]['data'][means][start_idx:start_idx + series_len]
        break
            
      except Exception as e:
          print(f"{e}. Retrying with a different sample...")
          series_len = None
          start_idx = None
          continue


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
    
    tot_ts_len = len(json_data[patient_id]["data"]["heart rate"])
    train_ts_len = int(0.8 * tot_ts_len)
    
    if series_len is None:
      series_len = random.randint(5, min(150, 5+int(train_ts_len/8)))
    if start_idx is None:
      if is_train:
        start_idx = random.randint(0, train_ts_len - series_len)
      else:
        start_idx = random.randint(train_ts_len, tot_ts_len - series_len)

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
    while True:
      try:
        country_ID = random.choice(list(json_data.keys()))
        attribute = random.choice([key for key in json_data[country_ID].keys() if key != "metadata"])
        
        tot_ts_len = len(json_data[country_ID][attribute])
        train_ts_len = int(0.8 * tot_ts_len)

        if series_len is None:
          series_len = random.randint(5, min(train_ts_len, tot_ts_len))
        if start_idx is None:
          if is_train:
            start_idx = random.randint(0, train_ts_len - series_len)
          else:
            start_idx = random.randint(train_ts_len, tot_ts_len - series_len)

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
        break
      
      except Exception as e:
        print(f"{e}")
        series_len = None
        start_idx = None
        continue


  elif dataset_name == "road injuries":
    while True:
      try:
        location = random.choice(list(json_data.keys()))
        mode = random.choice(list(json_data[location]['data'].keys()))
        severity = random.choice(list(json_data[location]['data'][mode].keys()))

        while len(json_data[location]['data'][mode][severity]) < 3: # if the series is too short, draw again
          location = random.choice(list(json_data.keys()))
          mode = random.choice(list(json_data[location]['data'].keys()))
          severity = random.choice(list(json_data[location]['data'][mode].keys()))

        tot_ts_len = len(json_data[location]['data'][mode][severity])
        train_ts_len = int(0.8 * tot_ts_len)
        #print(f"tot {tot_ts_len}, train_ts {train_ts_len}, series len {series_len}, start_idx {start_idx}")
        
        if series_len is None:
          series_len = random.randint(3, tot_ts_len - train_ts_len)
        if start_idx is None:
          if is_train:
            start_idx = random.randint(0, train_ts_len - series_len)
          else:
            start_idx = random.randint(train_ts_len, tot_ts_len - series_len)

        
        metadata = {}

        metadata['location'] = location
        metadata['mode'] = mode
        metadata['severity'] = severity
        metadata['geotype'] = json_data[location]["metadata"]['geotype']
        
        metadata['starting year'] = json_data[location]["metadata"]['start year of the series'] + start_idx
        metadata['end year'] = metadata['starting year'] + series_len - 1
        metadata['total population'] = json_data[location]['metadata']['totalpop']
        metadata['sampling frequency'] = "yearly"
        
        
        ts = json_data[location]['data'][mode][severity][start_idx:start_idx+series_len]
        ts = [round(x, 2) for x in ts]
        
        sum_series = np.zeros(series_len)
        count_series = np.zeros(series_len)
        for loc in json_data:
            if json_data[loc]["metadata"]["geotype"] == metadata["geotype"]:
                if mode in json_data[loc]['data'] and severity in json_data[loc]['data'][mode]:
                    series = json_data[loc]['data'][mode][severity]
                    for i, value in enumerate(series[:series_len]):  # Only consider existing values
                        sum_series[i] += value
                        count_series[i] += 1  # Count only non-padded values

        # Avoid division by zero by using np.where
        average_ts = np.where(count_series > 0, sum_series / count_series, np.nan)

        metadata['average time series of this type of location'] = [float(round(x, 2)) for x in average_ts]
        metadata['standard deviation of this type of location'] = float(round(np.std(metadata['average time series of this type of location']), 2))

        metadata['mean of this specific series'] = float(round(np.mean(ts), 2))
        metadata['standard deviation of this specific series'] = float(round(np.std(ts), 2))
        metadata['minimum of this specific series'] = float(round(min(ts), 2))
        metadata['maximum of this specific series'] = float(round(max(ts), 2))
        break
      except Exception as e:
        print(f"{e}")
        series_len = None
        start_idx = None
        continue

  if dataset_name == "covid":
    while True:
      try:
        country_ID = random.choice(list(json_data.keys()))
        country = json_data[country_ID]['metadata']['country_name']
        attribute = random.choice(list(json_data[country_ID].keys())) # daily_cases, daily_deaths
        while attribute == "metadata":
          attribute = random.choice(list(json_data[country_ID].keys()))
        
        tot_ts_len = len(json_data[country_ID][attribute])
        train_ts_len = int(0.8 * tot_ts_len)
          
        if series_len is None:
          series_len = random.randint(5, min(150, 5+int(train_ts_len/5)))
        if start_idx is None:
          if is_train:
            start_idx = random.randint(0, train_ts_len - series_len)
          else:
            start_idx = random.randint(train_ts_len, tot_ts_len - series_len)
        
        #print(f"tot len {tot_ts_len}, train len {train_ts_len}, series len {series_len}, start idx {start_idx}")
        ts = json_data[country_ID][attribute][start_idx:start_idx+series_len]
        ts = [round(x, 2) for x in ts]
        
        while (ts.count(0) / len(ts)) * 100 >= 20: # if there are at last 20% zeros, reject it and resample
          country_ID = random.choice(list(json_data.keys()))
          country = json_data[country_ID]['metadata']['country_name']
          attribute = random.choice(list(json_data[country_ID].keys())) # daily_cases, dauly_deaths
          while attribute == "metadata":
            attribute = random.choice(list(json_data[country_ID].keys()))
          
          tot_ts_len = len(json_data[country_ID][attribute])
          train_ts_len = int(0.8 * tot_ts_len)
            
          if series_len is None:
            series_len = random.randint(5, min(150, 5+int(train_ts_len/5)))
          if is_train:
            start_idx = random.randint(0, train_ts_len - series_len)
          else:
            start_idx = random.randint(train_ts_len, tot_ts_len - series_len)
          ts = json_data[country_ID][attribute][start_idx:start_idx+series_len]
          ts = [round(x, 2) for x in ts]
        
        start_date = pd.to_datetime(json_data[country_ID]['metadata']['start_date']) + pd.DateOffset(days=start_idx)
        end_date = pd.to_datetime(json_data[country_ID]['metadata']['start_date']) + pd.DateOffset(days=start_idx+series_len-1)
        start_date = start_date.strftime('%Y-%m-%d')
        end_date = end_date.strftime('%Y-%m-%d')
        break
      except Exception as e:
        print(f"{e}...")
        series_len = original_series_len
        start_idx = None
        continue
    
    metadata = {}
    metadata['country'] = country
    metadata['attribute'] = attribute
    metadata['sampling frequency'] = "daily"
    metadata['start date of this series'] = start_date
    metadata['end date of this series'] = end_date
    metadata['income group'] = json_data[country_ID]['metadata']['income_group'] # middle income
    metadata['region'] = json_data[country_ID]['metadata']['region'] # south asia
    if 'gdp_per_capita' in json_data[country_ID]['metadata']['stats']:
      metadata['gdp per capita'] = json_data[country_ID]['metadata']['stats']['gdp_per_capita']
    if 'aged_65_over' in json_data[country_ID]['metadata']['stats']:
      metadata['over 65'] = json_data[country_ID]['metadata']['stats']['aged_65_over']
    if 'median_age' in json_data[country_ID]['metadata']['stats']:
      metadata['median age'] = json_data[country_ID]['metadata']['stats']['median_age']
    if 'population_density' in json_data[country_ID]['metadata']['stats']:
      metadata['population density'] = json_data[country_ID]['metadata']['stats']['population_density']
    
    metadata['population'] = json_data[country_ID]['metadata']['population']
    metadata['historical minimum in this country'] = float(round(json_data[country_ID]['metadata']['stats']['min'][attribute]))
    metadata['historical maximum in this country'] = float(round(json_data[country_ID]['metadata']['stats']['max'][attribute]))
    metadata['historical mean in this country'] = float(round(json_data[country_ID]['metadata']['stats']['mean'][attribute]))
    metadata['historical standard deviation in this country'] = float(round(json_data[country_ID]['metadata']['stats']['std'][attribute]))
    
    metadata['mean of this specific series'] = float(round(np.mean(ts), 2))
    metadata['standard deviation of this specific series'] = float(round(np.std(ts), 2))
    metadata['minimum of this specific series'] = float(round(min(ts), 2))
    metadata['maximum of this specific series'] = float(round(max(ts), 2))
    
    
    
  if dataset_name == "co2":
    while True:
      try:
        country_ID = random.choice(list(json_data.keys()))
        country = json_data[country_ID]['metadata']['country_name']
        attribute = "co2_emissions"
          
        while True:
          try:
            tot_ts_len = len(json_data[country_ID][attribute])
            train_ts_len = int(0.8 * tot_ts_len)
            
            if series_len is None:
              series_len = random.randint(5, min(150, 5 + int(train_ts_len / 8)))
            if start_idx is None:
              if is_train:
                start_idx = random.randint(0, train_ts_len - series_len)
              else:
                start_idx = random.randint(train_ts_len, tot_ts_len - series_len)
            ts = json_data[country_ID][attribute][start_idx:start_idx + series_len]
            ts = [round(x, 2) for x in ts]
            break
          except Exception as e:
            print(f"Error occurred: {e}. Retrying...")
        
        start_year = json_data[country_ID]['metadata']['years'][start_idx]
        end_year = json_data[country_ID]['metadata']['years'][start_idx+series_len-1]
        
        metadata = {}
        metadata['country'] = country
        metadata['attribute'] = attribute
        metadata['sampling frequency'] = "yearly"
        metadata['start year of this series'] = start_year
        metadata['end year of this series'] = end_year
        metadata['region'] = json_data[country_ID]['metadata']['region'] # south asia

        metadata['population at the start year'] = json_data[country_ID]['population'][start_idx]
        metadata['population at the end year'] = json_data[country_ID]['population'][start_idx+series_len-1]
        
        #metadata['historical minimum in this country'] = float(round(json_data[country_ID]['metadata']['stats']['min'][attribute]))
        #metadata['historical maximum in this country'] = float(round(json_data[country_ID]['metadata']['stats']['max'][attribute]))
        #metadata['historical mean in this country'] = float(round(json_data[country_ID]['metadata']['stats']['mean'][attribute]))
        #metadata['historical standard deviation in this country'] = float(round(json_data[country_ID]['metadata']['stats']['std'][attribute]))
        
        metadata['mean of this specific series'] = float(round(np.mean(ts), 2))
        metadata['standard deviation of this specific series'] = float(round(np.std(ts), 2))
        metadata['minimum of this specific series'] = float(round(min(ts), 2))
        metadata['maximum of this specific series'] = float(round(max(ts), 2))
        break
      except Exception as e:
        print(f"{e}")
        series_len = None
        start_idx = None
        continue


  if dataset_name == "diet":
    valid_keys = list(json_data.keys())

    while True:
        try:
            # Randomly choose a country and attribute with enough data
            country_ID = random.choice(valid_keys)
            country_data = json_data[country_ID]
            country = country_data['metadata']['country_name']
            
            attributes = list(country_data["time_series"].keys())
            random.shuffle(attributes)  # Shuffle to reduce repeat attempts

            found_valid_series = False

            for attribute in attributes:
                series = country_data["time_series"][attribute]
                series_length = len(series)
                
                # Calculate train/test split
                train_ts_len = int(0.8 * series_length)

                # We need at least 15 data points to allow min series_len of 5 and margin of 10
                if is_train == False:
                  min_len = 5
                else:
                  min_len=10 
                  
                if series_length >= min_len:
                    if series_len is None:
                      if is_train==False:
                        series_len = random.randint(5, min(6, series_length-train_ts_len))
                      else:
                        series_len = random.randint(5, min(20, train_ts_len))
                    if start_idx is None:
                        if is_train:
                            max_start = train_ts_len - series_len
                        else:
                            max_start = series_length - series_len
                            min_start = train_ts_len
                            
                        if max_start <= 0 or (not is_train and train_ts_len >= series_length - series_len):
                            continue  # Not enough room to sample
                            
                        if is_train:
                            start_idx = random.randint(0, max_start)
                        else:
                            start_idx = random.randint(min_start, max_start)

                    ts = series[start_idx:start_idx + series_len]
                    start_year = country_data['metadata']['years'][start_idx]
                    end_year = int(start_year) + series_len - 1

                    # Redraw if too many zeroes
                    if (ts.count(0) / len(ts)) >= 0.2:
                        continue

                    found_valid_series = True
                    break

            if not found_valid_series:
                continue  # Try a different country

            break  # Valid sample found

        except Exception as e:
            print(f"Error occurred: {e}. Retrying...")

    # Final processing
    ts = [round(x, 2) for x in ts]

    metadata = {
        'country': country,
        'attribute': attribute,
        'sampling frequency': "yearly",
        'start year of this series': start_year,
        'end year of this series': end_year,
        'historical minimum in this country': float(round(country_data['metadata']['stats'][attribute]['min'])),
        'historical maximum in this country': float(round(country_data['metadata']['stats'][attribute]['max'])),
        'historical mean in this country': float(round(country_data['metadata']['stats'][attribute]['mean'])),
        'mean of this specific series': float(round(np.mean(ts), 2)),
        'minimum of this specific series': float(round(min(ts), 2)),
        'maximum of this specific series': float(round(max(ts), 2))
    }
 
    
    
  if dataset_name == "online retail":     
    while True:
      try:
        transaction_ID = random.choice(list(json_data.keys()))
        item = json_data[transaction_ID]["metadata"]['description']
        country = "United Kingdom"
        attribute = random.choice(list(json_data[transaction_ID]["time_series"]))
        
        while attribute == "dates": # redraw a sample if "dates" has been picked
          attribute = random.choice(list(json_data[transaction_ID]["time_series"]))
          
        tot_ts_len = len(json_data[transaction_ID]["time_series"][attribute])
        train_ts_len = int(0.8 * tot_ts_len)
          
        if series_len is None:
          if is_train:
            series_len = random.randint(5, train_ts_len)
          else:
            series_len = random.randint(5, tot_ts_len-train_ts_len)
        if start_idx is None:
          if is_train:
            start_idx = random.randint(0, train_ts_len - series_len)
          else:
            start_idx = random.randint(train_ts_len, tot_ts_len - series_len)
        ts = json_data[transaction_ID]['time_series'][attribute][start_idx:start_idx+series_len]
        ts = [round(x, 2) for x in ts]
        break
      except Exception as e:
        print(f"Error occurred: {e}. Retrying...")
        
    start_week = json_data[transaction_ID]['time_series']['dates'][start_idx]
    end_week = json_data[transaction_ID]['time_series']['dates'][start_idx+series_len-1]
    
    metadata = {}
    metadata['item'] = item
    metadata['country'] = country
    metadata['attribute'] = attribute
    metadata['sampling frequency'] = "weekly"
    metadata['start week of this series'] = start_week
    metadata['end week of this series'] = end_week
    
    metadata['average weekly customers'] = json_data[transaction_ID]['metadata']["stats"]["engagement"]['avg_weekly_customers']
    metadata['max sales'] = json_data[transaction_ID]['metadata']["stats"]["financial"]['max_sales_week']

    metadata['mean of this specific series'] = float(round(np.mean(ts), 2))
    metadata['minimum of this specific series'] = float(round(min(ts), 2))
    metadata['maximum of this specific series'] = float(round(max(ts), 2))
  
  if dataset_name == "walmart":  
      while True:
          try:
              ID = random.choice(list(json_data.keys()))
              sampling_frequency = json_data[ID]['metadata']['sampling_frequency']
              attribute = "weekly_sales"

              tot_ts_len = len(json_data[ID]["time_series"][attribute])
              train_ts_len = int(0.8 * tot_ts_len)

              if series_len is None:
                  series_len = random.randint(5, min(150, 5 + int(train_ts_len / 8)))

              if start_idx is None:
                  if is_train:
                      start_idx = random.randint(0, train_ts_len - series_len)
                  else:
                      start_idx = random.randint(train_ts_len, tot_ts_len - series_len)

              ts = json_data[ID]['time_series'][attribute][start_idx:start_idx + series_len]
              ts = [round(x, 2) for x in ts]
              break
          except Exception as e:
              print(f"Error occurred: {e}. Retrying...")

      start_week = json_data[ID]['time_series']['dates'][start_idx]
      end_week = json_data[ID]['time_series']['dates'][start_idx + series_len - 1]

      metadata = {}
      metadata['attribute'] = attribute
      metadata['sampling frequency'] = sampling_frequency
      metadata['start week of this series'] = start_week
      metadata['end week of this series'] = end_week

      metadata['best week sales'] = json_data[ID]['metadata']['stats']['sales']['best_week_sales']
      metadata['best week'] = json_data[ID]['metadata']['stats']['sales']['best_week']
      metadata['worst week sales'] = json_data[ID]['metadata']['stats']['sales']['worst_week_sales']
      metadata['worst week'] = json_data[ID]['metadata']['stats']['sales']['worst_week']
      metadata['mean sales'] = round(float(json_data[ID]['metadata']['stats']['sales']['mean']), 2)

      metadata['mean of this specific series'] = float(round(np.mean(ts), 2))
      metadata['minimum of this specific series'] = float(round(min(ts), 2))
      metadata['maximum of this specific series'] = float(round(max(ts), 2))

  if dataset_name == "agriculture":
      while True:
          try:
              country = random.choice(list(json_data.keys()))
              sampling_frequency = "yearly"
              attribute = random.choice(list(json_data[country]))
              while attribute in ["metadata", "years", "Output quantity"]:
                  attribute = random.choice(list(json_data[country]))

              tot_ts_len = len(json_data[country][attribute])
              train_ts_len = int(0.8 * tot_ts_len)

              if series_len is None:
                  series_len = random.randint(5, min(150, 5 + int(train_ts_len / 8)))

              if start_idx is None:
                  if is_train:
                      start_idx = random.randint(0, train_ts_len - series_len)
                  else:
                      start_idx = random.randint(train_ts_len, tot_ts_len - series_len)

              ts = json_data[country][attribute][start_idx:start_idx + series_len]
              ts = [round(x, 2) for x in ts]
              break
          except Exception as e:
              print(f"Error occurred: {e}. Retrying...")

      start_year = json_data[country]['years'][start_idx]
      end_year = json_data[country]['years'][start_idx + series_len - 1]

      metadata = {}
      metadata['country'] = country
      metadata['attribute'] = attribute
      metadata['sampling frequency'] = sampling_frequency
      metadata['start year of this series'] = start_year
      metadata['end year of this series'] = end_year

      metrics_info = json_data[country]['metadata']['metrics_definition'][attribute]
      if "components" in metrics_info:
          info = f"The {attribute} comprises the following components: {', '.join(metrics_info['components'])}."
      elif "calculation" in metrics_info:
          info = f"The {attribute} is computed as the {metrics_info['calculation']}."

      metadata['metrics info'] = info
      metadata['historical max'] = round(float(max(json_data[country][attribute])), 2)
      metadata['historical min'] = round(float(min(json_data[country][attribute])), 2)
      metadata['historical mean'] = round(float(np.mean(json_data[country][attribute])), 2)

      metadata['mean of this specific series'] = float(round(np.mean(ts), 2))
      metadata['minimum of this specific series'] = float(round(min(ts), 2))
      metadata['maximum of this specific series'] = float(round(max(ts), 2))

  return metadata, ts
  
  
# the following function does not preclude that no sample is duplicated, there's a very slim chance that it occurs
def get_samples(dataset_name, json_data, n, is_train, series_len=None) -> list: # returns a list of tuples (metadata, ts) of the specified dataset
  samples = []
  if n is not None: # this fixes the number of samples
    i = 0
    while i < n:
      metadata, ts = get_sample(dataset_name, json_data, is_train, series_len=series_len)
      if not np.isnan(ts).any() and not any(isinstance(x, str) and x.lower() == 'nan' for x in ts):
        zero_percentage = (ts.count(0) / len(ts)) * 100
        if zero_percentage <= 10:
            samples.append((metadata, ts))
            i += 1
      
  return samples

def get_request(dataset_name, metadata, ts, external_knowledge=False):
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
  elif dataset_name == "road injuries":
    request = f"""I will give you a time series about the {metadata['sampling frequency']} number of people getting {metadata['severity']} on the road by means of {metadata['mode']}. The location is the {metadata['geotype']} of {metadata['location']}  and the period is from {metadata['starting year']} to {metadata['end year']}.
          {metadata['location']} has a total population of {metadata['total population']}.
           Here is the time series: \n {ts}
          \nHere are the statistics for this specific time series for {metadata['location']} from {metadata['starting year']} to {metadata['end year']}: \n Mean: {metadata['mean of this specific series']} \n Standard Deviation: {metadata['standard deviation of this specific series']} \n Minimum: {metadata['minimum of this specific series']} \n Maximum: {metadata['maximum of this specific series']}
          \nHere is the average time series of a typical {metadata['geotype']} in California in the same period of years. \n {metadata['average time series of this type of location']}, whose standard deviation is {metadata['standard deviation of this type of location']}

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
  elif dataset_name == "covid":
    request = f"""I will give you a time series about the {metadata['sampling frequency']} {metadata['attribute']} due to Covid 19 in the country of {metadata['country']}, {metadata['region']}. The country has a population of {metadata['population']} and is classified as {metadata['income group']}. {f"The country has a GDP per capita of {metadata['gdp per capita']}." if 'gpt per capita' in metadata.keys() else ""}
    {f"The percentage of over 65 is {metadata['over 65']}." if 'over 65' in metadata.keys() else ""} {f"The median age in the country is {metadata['median age']}." if 'median age' in metadata.keys() else ""} {f"The country has a population density of {metadata['population density']}." if 'population density' in metadata.keys() else ""}     
    
    The time series covers the period from {metadata['start date of this series']} to {metadata['end date of this series']}.
    
           Here is the time series: \n {ts}
           
          \nHere are the statistics for this specific time series for {metadata['country']} from {metadata['start date of this series']} to {metadata['end date of this series']}: \n Mean: {metadata['mean of this specific series']} \n Standard Deviation: {metadata['standard deviation of this specific series']} \n Minimum: {metadata['minimum of this specific series']} \n Maximum: {metadata['maximum of this specific series']}
          \nHere are the general statistics about the  {metadata['sampling frequency']} {metadata['attribute']} in {metadata['country']}. \n Mean: {metadata['historical mean in this country']} \n Standard Deviation: {metadata['historical standard deviation in this country']} \n Minimum: {metadata["historical minimum in this country"]} \n Maximum: {metadata["historical maximum in this country"]}

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
  elif dataset_name == "co2":
    request = f"""I will give you a time series about the {metadata['sampling frequency']} co2 emissions measured in million metric tons, in the country of {metadata['country']}, located in {metadata['region']}. 
    
    The time series covers the period from {metadata['start year of this series']} to {metadata['end year of this series']}, with the national population of {metadata['population at the start year']} and {metadata['population at the end year']} respectively.
    
    Here is the time series: \n {ts}
           
          \nHere are the statistics for this specific time series for {metadata['country']} from {metadata['start year of this series']} to {metadata['end year of this series']}: \n Mean: {metadata['mean of this specific series']} \n Standard Deviation: {metadata['standard deviation of this specific series']} \n Minimum: {metadata['minimum of this specific series']} \n Maximum: {metadata['maximum of this specific series']}

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
  elif dataset_name == "diet":
    request = f"""I will give you a time series about the {metadata['sampling frequency']} average per capita daily kilocalories consumed from {metadata['attribute']} in the country of {metadata['country']}. 
    
    The time series covers the period from {metadata['start year of this series']} to {metadata['end year of this series']}.
    Here is the time series: \n {ts}
           
          \nHere are the statistics for this specific time series for {metadata['country']} from {metadata['start year of this series']} to {metadata['end year of this series']}: \nMean: {metadata['mean of this specific series']} \nMinimum: {metadata['minimum of this specific series']} \nMaximum: {metadata['maximum of this specific series']}
          
          \nHere are the all-time statistics of the {metadata["attribute"]} in {metadata['country']}, until the present year. \nAll-time minimum: {metadata['historical minimum in this country']}\nAll-time maximum: {metadata['historical maximum in this country']} \nAll-time mean: {metadata['historical mean in this country']}

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
  elif dataset_name == "online retail":
    request = f"""I will give you a time series about the {metadata['sampling frequency']} {metadata['attribute'].replace("_", " ")} of the item: "{metadata['item']}" from an online retailer in the  {metadata['country']}. 
    
    The time series covers the period from the week of {metadata['start week of this series']} to the week of {metadata['end week of this series']}.
    Here is the time series expressed in GBP: \n {ts}
           
          \nHere are the statistics for this specific time series for {metadata['item']}. \nMean: {metadata['mean of this specific series']} \nMinimum: {metadata['minimum of this specific series']} \nMaximum: {metadata['maximum of this specific series']}
          
          \nHere are some additional information: \nAverage weekly customers of this item: {metadata['average weekly customers']} \nMaximum weekly sales ever: {metadata['max sales']} GBP.

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
          
  elif dataset_name == "walmart":
    request = f"""I will give you a time series about the {metadata['sampling frequency']} {metadata['attribute'].replace("_", " ")} of a Walmart store, from the week of {metadata['start week of this series']} to the week of {metadata['end week of this series']}.
    Here is the time series expressed in USD: \n {ts}
           
          \nHere are the statistics for this specific time series. \nMean: {metadata['mean of this specific series']} \nMinimum: {metadata['minimum of this specific series']} \nMaximum: {metadata['maximum of this specific series']}
          
          \nHere are some additional information: \nBest ever weekly sales: {metadata['best week sales']} USD on the week of {metadata['best week']} \nWorst ever weekly sales: {metadata['worst week sales']} USD on the week of {metadata['worst week']} \nMean sales between 2010 and 2012: {metadata['mean sales']}

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
  elif dataset_name == "agriculture":
    request = f"""I will give you a time series about the {metadata['sampling frequency']} {metadata['attribute']} in the country of {metadata['country']}, from {metadata['start year of this series']} to {metadata['end year of this series']}. {metadata['metrics info']}
    Here is the time series: \n {ts}
           
          \nHere are the statistics for this specific time series. \nMean: {metadata['mean of this specific series']} \nMinimum: {metadata['minimum of this specific series']} \nMaximum: {metadata['maximum of this specific series']}
          
          \nHere are some additional information until 2019: \nHistorical maximum: {metadata['historical max']} \nHistorical minimum: {metadata['historical min']} \nHistorical mean: {metadata['historical mean']}

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
                file.write(str(item) + '\n')
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

def change_linguistic_style(caption, style="casual", model="Google Gemini 2.0 Flash"):
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

def generate_line_plot(
    ts, xlabel, ylabel, title, savepath,
    height=None, width=None, color="blue",
    linewidth=1, marker='o', linestyle='-',
    grid=False, show_nums_on_line=False,
    x_start=None, x_end=None
):
    # Set figure size if specified
    figsize = (width, height) if width and height else None
    plt.figure(figsize=figsize)

    # Plot the time series with the specified style
    plt.plot(
        ts,
        color=color,
        linewidth=linewidth,
        marker=marker,
        linestyle=linestyle
    )
    
    if x_start is not None and x_end is not None:
        plt.xticks([0, len(ts) - 1], [x_start, x_end])
        
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(grid)
    
    fs = random.randint(7, 12)
    if show_nums_on_line:
      if len(ts) < 25:
        for i, val in enumerate(ts):
          plt.text(i, val, f'{val}', ha='center', va='bottom', fontsize=fs)

    # Save and close the plot
    plt.tight_layout()
    plt.savefig(savepath, dpi=100)
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

def bert_score(bert_model, tokenizer, generated_captions, gt_captions):
    """
    Compute the BERT score loss for the generated captions compared to the ground-truth captions.
    The loss is based on minimizing the cosine similarity between the BERT embeddings of the generated
    and ground-truth captions.

    Args:
        bert_model: Pretrained BERT model (should be initialized outside this function).
        generated_captions (list of str): List of generated captions.
        gt_captions (list of str): List of ground-truth captions.

    Returns:
        score (tensor): average cosine similarity between generated and ground-truth embeddings
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tokenize and process both generated and ground-truth captions
    generated_input = tokenizer(generated_captions, return_tensors="pt", padding=True, truncation=True).to(device)
    gt_input = tokenizer(gt_captions, return_tensors="pt", padding=True, truncation=True).to(device)

    # Get embeddings from BERT for both generated and ground-truth captions
    with torch.no_grad():  # Freeze BERT, no gradients will be computed for BERT
        generated_embeddings = bert_model(**generated_input).last_hidden_state.mean(dim=1)
        gt_embeddings = bert_model(**gt_input).last_hidden_state.mean(dim=1)

    # Normalize the embeddings (important for cosine similarity)
    generated_embeddings = F.normalize(generated_embeddings, dim=-1)
    gt_embeddings = F.normalize(gt_embeddings, dim=-1)

    # Compute the cosine similarity between the generated and ground-truth embeddings
    cosine_sim = F.cosine_similarity(generated_embeddings, gt_embeddings, dim=-1)

    score = cosine_sim.mean()

    return score

def oracle_score(generated_caption, gt_caption, model="Google Gemini-2.0-Flash"):
  prompt = f"""
  You are an expert evaluator of artificially generated captions against ground truth captions. 
  Your task is to provide scores (0-100) for the generated caption's quality in comparison to the ground truth.

  Scoring Criteria:
  1.  Semantic Similarity: How closely does the generated caption convey the same meaning as the ground truth?
  2.  Information Overlap: How much of the factual information present in the ground truth is also accurately represented in the generated caption?
  3.  Numeric Correctness: Are all numbers in the generated caption exactly the same as those in the ground truth? A single numerical mismatch results in a score of 0.
  4.  Overall Quality: A holistic score reflecting the overall accuracy and usefulness of the generated caption. Assign higher weight to numeric correctness and semantic similarity.

  Examples:
  Generated: "The chart shows a slight increase in sales."
  Ground Truth: "Sales increased by 5%."
  Scores:
  - Semantic Similarity: 80
  - Information Overlap: 50
  - Numeric Correctness: 0
  - Overall: 40

  Generated: "The average daily temperature of Seattle was 11 degrees Celsius in the beginning of March 2023, and it increased to 14 by the end of the month."
  Ground Truth: "The average daily temperature of Seattle was 10 degrees Celsius in the beginning of March 2023, and it increased to 14 by the end of the month."
  Scores:
  - Semantic Similarity: 100
  - Information Overlap: 100
  - Numeric Correctness: 50
  - Overall: 70

  Generated: "There are two peaks in this time series."
  Ground Truth: "The time series shows two distinct peaks."
  Scores:
  - Semantic Similarity: 95
  - Information Overlap: 100
  - Numeric Correctness: 100
  - Overall: 98

  Generated: "{generated_caption}"
  Ground Truth: "{gt_caption}"

  Provide your scores in the following STRICT format:
  - Semantic Similarity: [score]
  - Information Overlap: [score]
  - Numeric Correctness: [score]
  - Overall: [score]

  Do NOT include any additional text or explanations.
  """

  response = get_response(prompt, model=model, temperature=0.25)
  #print(f"Scoring response:\n{response}")
  response = response.lower()
  #semantic_similarity_score = int(re.search(r"semantic similarity: (\d+)", response).group(1))
  #information_overlap_score = int(re.search(r"information overlap: (\d+)", response).group(1))
  #numeric_correctness_score = int(re.search(r"numeric correctness: (\d+)", response).group(1))
  overall_score = int(re.search(r"overall: (\d+)", response).group(1))

  """return {
      "semantic similarity": semantic_similarity_score,
      "information overlap": information_overlap_score,
      "numeric correctness": numeric_correctness_score,
      "overall": overall_score
  }"""
  return overall_score

def generate_prompt_for_baseline(dataset_name, metadata, ts):
  config=load_config()
  external_knowledge = config['data']['external_knowledge']
  
  if dataset_name == "air quality":
    request = f"""Here is a time series about {metadata["sampling frequency"]} {metadata["measure"]} in the Indian city of {metadata['city']}: \n {ts} \n
    \n Describe this time series by focusing on trends and patterns. 
    Discuss concrete numbers you see and pay attention to the dates.
    For numerical values, ensure consistency with the provided time series. If making percentage comparisons, round to the nearest whole number. Report the dates when things happened.
          
    Compare the trends in this time series to global or regional norms, explaining whether they are higher, lower, or follow expected seasonal patterns.
    When making comparisons, clearly state whether differences are minor, moderate, or significant.
    Use descriptive language to create engaging, natural-sounding text.
    Avoid repetitive phrasing and overused expressions.

    Answer in a single paragraph of four sentences at most, without bullet points or any formatting.
    """

  elif dataset_name == "crime":
    request = f"""Here is a time series about the number of {metadata["sampling frequency"]} crimes {metadata["town"]}, Los Angeles, from {metadata["start date of the series"]} to {metadata["end date of the series"]}: \n {ts}

    \n Describe this time series by focusing on trends and patterns. 
    Discuss concrete numbers you see and pay attention to the dates.
    For numerical values, ensure consistency with the provided time series. If making percentage comparisons, round to the nearest whole number. Report the dates when things happened.
          
    Compare the trends in this time series to global or regional norms, explaining whether they are higher, lower, or follow expected seasonal patterns.
    When making comparisons, clearly state whether differences are minor, moderate, or significant.
    Use descriptive language to create engaging, natural-sounding text.
    Avoid repetitive phrasing and overused expressions.

    Answer in a single paragraph of four sentences at most, without bullet points or any formatting.
     """

  elif dataset_name == "border crossing":
    request = f"""Here is a time series about the number of {metadata['sampling frequency']} {metadata['means']} crossing the port of {metadata['port']} at the {metadata["border"]} border, starting from {metadata["start date of the series"]}Here is the time series: \n {ts}

    \n Describe this time series by focusing on trends and patterns. 
    Discuss concrete numbers you see and pay attention to the dates.
    For numerical values, ensure consistency with the provided time series. If making percentage comparisons, round to the nearest whole number. Report the dates when things happened.
          
    Compare the trends in this time series to global or regional norms, explaining whether they are higher, lower, or follow expected seasonal patterns.
    When making comparisons, clearly state whether differences are minor, moderate, or significant.
    Use descriptive language to create engaging, natural-sounding text.
    Avoid repetitive phrasing and overused expressions.

    Answer in a single paragraph of four sentences at most, without bullet points or any formatting.
     """


  elif dataset_name == "demography":
    request = f"""I will give you a time series about the {metadata['sampling frequency']} {metadata['attribute']} of {metadata['country']} from {metadata['starting year']} to {metadata['end year']}, it's measured as number per 1000 people.{metadata['country']} is categorized as a country with these attributes: {metadata['category by income']}.
    Here is the time series: \n {ts}

    \n Describe this time series by focusing on trends and patterns. 
    Discuss concrete numbers you see and pay attention to the dates.
    For numerical values, ensure consistency with the provided time series. If making percentage comparisons, round to the nearest whole number. Report the dates when things happened.
          
    Compare the trends in this time series to global or regional norms, explaining whether they are higher, lower, or follow expected seasonal patterns.
    When making comparisons, clearly state whether differences are minor, moderate, or significant.
    Use descriptive language to create engaging, natural-sounding text.
    Avoid repetitive phrasing and overused expressions.

    Answer in a single paragraph of four sentences at most, without bullet points or any formatting.
     """
  elif dataset_name == "road injuries":
    request = f"""I will give you a time series about the {metadata['sampling frequency']} number of people getting {metadata['severity']} on the road by means of {metadata['mode']}. The location is the {metadata['geotype']} of {metadata['location']}  and the period is from {metadata['starting year']} to {metadata['end year']}. {metadata['location']} has a total population of {metadata['total population']}.
    Here is the time series: \n {ts}

    \n Describe this time series by focusing on trends and patterns. 
    Discuss concrete numbers you see and pay attention to the dates.
    For numerical values, ensure consistency with the provided time series. If making percentage comparisons, round to the nearest whole number. Report the dates when things happened.
          
    Compare the trends in this time series to global or regional norms, explaining whether they are higher, lower, or follow expected seasonal patterns.
    When making comparisons, clearly state whether differences are minor, moderate, or significant.
    Use descriptive language to create engaging, natural-sounding text.
    Avoid repetitive phrasing and overused expressions.

    Answer in a single paragraph of four sentences at most, without bullet points or any formatting.
     """
  elif dataset_name == "covid":
    request = f"""I will give you a time series about the {metadata['sampling frequency']} {metadata['attribute']} due to Covid 19 in the country of {metadata['country']}, {metadata['region']}. The country has a population of {metadata['population']} and is classified as {metadata['income group']}. {f"The country has a GDP per capita of {metadata['gdp per capita']}." if 'gpt per capita' in metadata.keys() else ""}
    {f"The percentage of over 65 is {metadata['over 65']}." if 'over 65' in metadata.keys() else ""} {f"The median age in the country is {metadata['median age']}." if 'median age' in metadata.keys() else ""} {f"The country has a population density of {metadata['population density']}." if 'population density' in metadata.keys() else ""}

    The time series covers the period from {metadata['start date of this series']} to {metadata['end date of this series']}.

    Here is the time series: \n {ts}

    \n Describe this time series by focusing on trends and patterns. 
    Discuss concrete numbers you see and pay attention to the dates.
    For numerical values, ensure consistency with the provided time series. If making percentage comparisons, round to the nearest whole number. Report the dates when things happened.
          
    Compare the trends in this time series to global or regional norms, explaining whether they are higher, lower, or follow expected seasonal patterns.
    When making comparisons, clearly state whether differences are minor, moderate, or significant.
    Use descriptive language to create engaging, natural-sounding text.
    Avoid repetitive phrasing and overused expressions.

    Answer in a single paragraph of four sentences at most, without bullet points or any formatting.
     """
  elif dataset_name == "co2":
    request = f"""I will give you a time series about the {metadata['sampling frequency']} co2 emissions measured in million metric tons, in the country of {metadata['country']}, located in {metadata['region']}.

    The time series covers the period from {metadata['start year of this series']} to {metadata['end year of this series']}, with the national population of {metadata['population at the start year']} and {metadata['population at the end year']} respectively.

    Here is the time series: \n {ts}

   \n Describe this time series by focusing on trends and patterns. 
    Discuss concrete numbers you see and pay attention to the dates.
    For numerical values, ensure consistency with the provided time series. If making percentage comparisons, round to the nearest whole number. Report the dates when things happened.
          
    Compare the trends in this time series to global or regional norms, explaining whether they are higher, lower, or follow expected seasonal patterns.
    When making comparisons, clearly state whether differences are minor, moderate, or significant.
    Use descriptive language to create engaging, natural-sounding text.
    Avoid repetitive phrasing and overused expressions.

    Answer in a single paragraph of four sentences at most, without bullet points or any formatting.
     """
  elif dataset_name == "diet":
    request = f"""I will give you a time series about the {metadata['sampling frequency']} average per capita daily kilocalories consumed from {metadata['attribute']} in the country of {metadata['country']}.

    The time series covers the period from {metadata['start year of this series']} to {metadata['end year of this series']}.
    Here is the time series: \n {ts}

    \n Describe this time series by focusing on trends and patterns. 
    Discuss concrete numbers you see and pay attention to the dates.
    For numerical values, ensure consistency with the provided time series. If making percentage comparisons, round to the nearest whole number. Report the dates when things happened.
          
    Compare the trends in this time series to global or regional norms, explaining whether they are higher, lower, or follow expected seasonal patterns.
    When making comparisons, clearly state whether differences are minor, moderate, or significant.
    Use descriptive language to create engaging, natural-sounding text.
    Avoid repetitive phrasing and overused expressions.

    Answer in a single paragraph of four sentences at most, without bullet points or any formatting.
     """
  elif dataset_name == "online retail":
    request = f"""I will give you a time series about the {metadata['sampling frequency']} {metadata['attribute'].replace("_", " ")} of the item: "{metadata['item']}" from an online retailer in the  {metadata['country']}.

    The time series covers the period from the week of {metadata['start week of this series']} to the week of {metadata['end week of this series']}.
    Here is the time series expressed in GBP: \n {ts}

    \n Describe this time series by focusing on trends and patterns. 
    Discuss concrete numbers you see and pay attention to the dates.
    For numerical values, ensure consistency with the provided time series. If making percentage comparisons, round to the nearest whole number. Report the dates when things happened.
          
    Compare the trends in this time series to global or regional norms, explaining whether they are higher, lower, or follow expected seasonal patterns.
    When making comparisons, clearly state whether differences are minor, moderate, or significant.
    Use descriptive language to create engaging, natural-sounding text.
    Avoid repetitive phrasing and overused expressions.

    Answer in a single paragraph of four sentences at most, without bullet points or any formatting.
     """

  elif dataset_name == "walmart":
    request = f"""I will give you a time series about the {metadata['sampling frequency']} {metadata['attribute'].replace("_", " ")} of a Walmart store, from the week of {metadata['start week of this series']} to the week of {metadata['end week of this series']}.
    Here is the time series expressed in USD: \n {ts}

    \n Describe this time series by focusing on trends and patterns. 
    Discuss concrete numbers you see and pay attention to the dates.
    For numerical values, ensure consistency with the provided time series. If making percentage comparisons, round to the nearest whole number. Report the dates when things happened.
          
    Compare the trends in this time series to global or regional norms, explaining whether they are higher, lower, or follow expected seasonal patterns.
    When making comparisons, clearly state whether differences are minor, moderate, or significant.
    Use descriptive language to create engaging, natural-sounding text.
    Avoid repetitive phrasing and overused expressions.

    Answer in a single paragraph of four sentences at most, without bullet points or any formatting.
     """
  elif dataset_name == "agriculture":
    request = f"""I will give you a time series about the {metadata['sampling frequency']} {metadata['attribute']} in the country of {metadata['country']}, from {metadata['start year of this series']} to {metadata['end year of this series']}. {metadata['metrics info']}
    Here is the time series: \n {ts}

    \n Describe this time series by focusing on trends and patterns. 
    Discuss concrete numbers you see and pay attention to the dates.
    For numerical values, ensure consistency with the provided time series. If making percentage comparisons, round to the nearest whole number. Report the dates when things happened.
          
    Compare the trends in this time series to global or regional norms, explaining whether they are higher, lower, or follow expected seasonal patterns.
    When making comparisons, clearly state whether differences are minor, moderate, or significant.
    Use descriptive language to create engaging, natural-sounding text.
    Avoid repetitive phrasing and overused expressions.

    Answer in a single paragraph of four sentences at most, without bullet points or any formatting.
     """
  return request

def remove_years(nums):
    """
    Remove numbers that represent years from the list.
    
    A number is considered a year if:
    1. It's between 1960 and 2025 (inclusive)
    2. It doesn't have a non-zero decimal part (e.g., 1960.0 counts as a year, but 1960.1 doesn't)
    
    Args:
        nums: List of numbers (can be int or float)
        
    Returns:
        List of numbers with years removed
    """
    result = []
    
    for num in nums:
        # Check if the number is between 1960 and 2025
        if 1960 <= num <= 2025:
            # Check if it has no decimal part or if the decimal part is zero
            if isinstance(num, int) or num.is_integer():
                continue  # Skip this number as it's a year
        
        # If we got here, the number isn't a year, so add it to the result
        result.append(num)
    
    return result

def numeric_score(generated_caption, gt_caption, tolerance=0.05, acc_coef=0.3, recall_coef=0.7):
    """extract the non-temporal numbers from both and compare the numbers extracted from generated_caption to the number extracted from gt_caption
    check how good is their matching. If some numbers in gt are not in the generated_caption, that's a penalty. Also if some values are very close, it should be tolerated."""
    
    # Extract numbers from captions using regex
    # Pattern matches numbers including optional decimal point, but excludes temporal patterns like dates, times
    # Excluding temporal patterns like MM/DD/YYYY, HH:MM, etc.
    pattern = r'-?\d+(?:\.\d+)?'
    gen_numbers = [float(match) for match in re.findall(pattern, generated_caption)]
    gt_numbers = [float(match) for match in re.findall(pattern, gt_caption)]
    
    gen_numbers = remove_years(gen_numbers)
    gt_numbers = remove_years(gt_numbers)
    
    #print("gen numbers: ", gen_numbers)
    #print("gt numbers: ", gt_numbers)
    
    if not gt_numbers:
        # If no numbers in ground truth, return perfect score
        return 1.0
    
    if not gen_numbers:
        # If no numbers in generated caption but there are in ground truth, return 0
        return 0.0
    
    # Group similar numbers (accounting for different units possibly)
    # For each GT number, find the closest match in generated numbers
    matches = []
    unmatched_gt = []
    used_gen_indices = set()
       
    for gt_num in gt_numbers:
        best_match = None
        best_match_idx = -1
        best_relative_diff = float('inf')
        
        for i, gen_num in enumerate(gen_numbers):
            if i in used_gen_indices:
                continue
                
            # Calculate relative difference
            relative_diff = abs(gt_num - gen_num) / max(abs(gt_num), 1e-10)
            
            if relative_diff < best_relative_diff:
                best_relative_diff = relative_diff
                best_match = gen_num
                best_match_idx = i
        
        # If found a match within tolerance
        if best_match is not None and best_relative_diff <= tolerance:
            matches.append((gt_num, best_match, best_relative_diff))
            used_gen_indices.add(best_match_idx)
        else:
            unmatched_gt.append(gt_num)
    
    # Calculate score components
    recall = len(matches) / len(gt_numbers)  # How many GT numbers were matched
    
    # Calculate average accuracy of matches (how close were the matches)
    if matches:
        avg_accuracy = np.mean([1 - min(rel_diff, tolerance) / tolerance for _, _, rel_diff in matches])
    else:
        avg_accuracy = 0.0
    
    # Final score is a weighted combination of recall and accuracy
    score = acc_coef * avg_accuracy + recall_coef * recall
    
    return round(score, 2)

def bleu_score(generated_caption, gt_caption, n_gram_weights=(0.25, 0.25, 0.25, 0.25)): # n_gram_weights contains the weights for 1, 2, 3, and 4-grams.
    """
    Calculate BLEU score between a generated caption and ground truth caption.
    
    Args:
        generated_caption (str): The caption generated by a model
        gt_caption (str): The ground truth caption
        
    Returns:
        float: BLEU score between 0 and 1
    """
    # Tokenize the captions
    reference_tokens = [nltk.word_tokenize(gt_caption.lower())]
    candidate_tokens = nltk.word_tokenize(generated_caption.lower())
    
    # Apply smoothing to handle cases where n-grams don't match
    smoothie = SmoothingFunction().method1
    
    score = sentence_bleu(reference_tokens, 
                               candidate_tokens, 
                               weights=n_gram_weights, 
                               smoothing_function=smoothie)
    return round(score, 4)
  
def rouge_score(generated_caption, gt_caption, rouge_types=['rougeL'], metric="f1"):
    """
    Calculate ROUGE scores between a generated caption and ground truth caption.
    
    Uses the rouge-score package which is a Python implementation of ROUGE.
    
    Args:
        generated_caption (str): The caption generated by a model
        gt_caption (str): The ground truth caption
        rouge_types (list, optional): List of ROUGE types to calculate.
                                     Options: 'rouge1', 'rouge2', 'rougeL'
                                     If None, all three types are calculated.
        
    Returns:
        dict: Dictionary containing requested ROUGE scores (precision, recall, and F1)
    """
    # Default to all ROUGE types if none specified
    if rouge_types is None:
        rouge_types = ['rouge1', 'rouge2', 'rougeL']
    
    # Validate rouge_types
    valid_types = {'rouge1', 'rouge2', 'rougeL'}
    for r_type in rouge_types:
        if r_type not in valid_types:
            raise ValueError(f"Invalid ROUGE type: {r_type}. Valid options are: {valid_types}")
    
    # Initialize the ROUGE scorer with requested types
    scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)
    
    # Calculate scores
    scores = scorer.score(gt_caption, generated_caption)
    
    # Create a more readable output dictionary
    result = {}
    for rouge_type in rouge_types:
        result[rouge_type] = {
            'precision': scores[rouge_type].precision,
            'recall': scores[rouge_type].recall,
            'f1': scores[rouge_type].fmeasure
        }
    
    return result[rouge_type][metric]

def meteor_score(generated_caption, gt_caption):
    """
    Calculate the METEOR score between a generated caption and a ground truth caption.
    
    METEOR (Metric for Evaluation of Translation with Explicit ORdering) is a metric
    for evaluating machine translation output that is based on the harmonic mean of
    unigram precision and recall, with recall weighted higher than precision.
    
    Args:
        generated_caption (str): The caption generated by a model
        gt_caption (str): The ground truth caption
        
    Returns:
        float: METEOR score between 0 and 1
    """
    """# Download required NLTK resources if not already downloaded
    try:
        # Check if wordnet is available
        nltk.data.find('wordnet')
    except LookupError:
        print("Downloading wordnet...")
        nltk.download('wordnet')
    
    try:
        # Check if punkt is available
        nltk.data.find('punkt')
    except LookupError:
        print("Downloading punkt...")
        nltk.download('punkt')"""
    
    # Tokenize the captions
    reference_tokens = [gt_caption.split()]  # METEOR expects a list of references
    candidate_tokens = generated_caption.split()
    
    # Calculate METEOR score
    score = meteor_sc(reference_tokens, candidate_tokens)
    
    return score
 
def create_metric_comparisons(data_ft, data_base, model_name, save_path="/home/ubuntu/thesis/source/figs/", label="percentage"):
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Metrics and columns from the data
    metric_groups = [
        "BERT F1", "BERT Precision", "BERT Recall", "Numeric Score",
        "BLEU", "ROUGE-L", "METEOR", "Oracle Score", "simCSE"
    ]

    # Columns (categories)
    columns = [
        "Average", "Air Quality", "Border Crossing", "Crime", "Demography",
        "Road Injuries", "Covid", "Co2", "Diet", "Walmart", "Online Retail", "Agriculture"
    ]

    # Create grouped bar plots
    for col_idx, col_name in enumerate(columns):
        x = np.arange(len(metric_groups))  # positions for metric groups
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 5))
        exp1_vals = [data_ft[i][col_idx] for i in range(len(metric_groups))]
        exp2_vals = [data_base[i][col_idx] for i in range(len(metric_groups))]
        
        ax.bar(x - width/2, exp1_vals, width, label='Finetuned (LLaVA-v1.6-7B FT)', color="red")
        ax.bar(x + width/2, exp2_vals, width, label='Pretrained (LLaVA-v1.6-7B)', color='blue')

        # Compute dynamic y-limit to leave space for annotations
        max_val = max(max(exp1_vals), max(exp2_vals))
        ax.set_ylim(0, max_val * 1.15)  # 15% headroom

        if label == "percentage":
          for i, (v1, v2) in enumerate(zip(exp1_vals, exp2_vals)):
              if v2 != 0:
                  rel_improvement = ((v1 - v2) / v2) * 100
                  label = f"{rel_improvement:+.1f}%"
              else:
                  label = "N/A"
              mid_x = x[i]
              top = max(v1, v2)
              offset = 0.02 * max_val if max_val > 1 else 0.02
              ax.text(mid_x, top + offset, label, ha='center', va='bottom', fontsize=8, color='black')
        
        elif label == "delta":
          for i, (v1, v2) in enumerate(zip(exp1_vals, exp2_vals)):
              delta = v1 - v2
              mid_x = x[i]
              top = max(v1, v2)
              ax.text(mid_x, top + 0.02 * (1 if top < 1 else top), f"{delta:+.3f}", ha='center', va='bottom', fontsize=8, color='black')
        
        ax.set_ylabel('Score')
        ax.set_title(f'{model_name} Metrics Comparison - {col_name}')
        ax.set_xticks(x)
        ax.set_xticklabels(metric_groups, rotation=45, ha='right')
        ax.legend()
        plt.tight_layout()
        plt.grid(True)
        plt.savefig(f"{save_path}{model_name}_{col_name}_metrics.jpeg")
        plt.show()

def extract_numbers_with_semantic_context(text, ignore_dates=True):
    doc = nlp(text)
    number_contexts = []
    
    for token in doc:
      if token.like_num and token.is_digit and 1960 <= int(token.text) <= 2025: # skip year numbers
        continue
      if token.like_num:
          # Find the semantic head this number is modifying
          if token.head != token:
              context = token.head.text
              # Expand to include other modifiers of the head
              for child in token.head.children:
                  if child.dep_ in ['amod', 'compound', 'nmod']:
                      context = f"{child.text} {context}"
              # Perform stemming on the context
              
              context = ' '.join([stemmer.stem(word) for word in context.split()])
              number_contexts.append((token.text, context))
  
    return number_contexts

def extract_num_dict_from_text(caption, model="Google Gemini-2.0-Flash"):
  prompt = f"""
    I will provide you with a paragraph of text, which described a time series. Your job is to extract important numbers from it. Specifically, you have to extract minimum, maximum, mean, and standard deviation of this specific time series (not historical statistics) from the text if the text mentions them.
    
    Here's the text:
    \n
    {caption}
    \n
    
    Provide your answer in a json format as follows:
   {{
      "minimum": A,
      "maximum": B,
      "mean": X,
      "std": Y
    
   }}
   If you cannot find some variables in the text because they are not mentioned, just put null instead. Provide your answer in a json format and do not say anything more and don't give any explanation. Start your answer directly by opening a brace.
  
  """
  response = get_response(prompt=prompt, model=model, temperature=0.25)
  response = response[response.find("{"):response.rfind("}") + 1]
  try:
    response_dict = json.loads(response)
    return response_dict
  except Exception as e:
    print(e)
    print(f"Cannot be parsed!\n {response}")


def extract_num_dict_from_dict(metadata, model="Google Gemini-2.0-Flash"):
  prompt = f"""
    I will provide you with a json dictionary consisting of some metadata of a time series. Your job is to extract some numbers from it. Specifically, you have to extract minimum, maximum, mean, and standard deviation of this specific time series.
    
    Here's the dictionary:
    \n
    {metadata}
    \n
    
    Provide your answer in a json format as follows:
   {{
      "minimum": A,
      "maximum": B,
      "mean": X,
      "std": Y
    
   }}
   Provide your answer in a json format and do not say anything more and don't give any explanation. Start your answer directly by opening a brace.
  
  """
  response = get_response(prompt=prompt, model=model, temperature=0.25)
  response = response[response.find("{"):response.rfind("}") + 1]
  try:
    response_dict = json.loads(response)
    return response_dict
  except Exception as e:
    print(e)
    print(f"Cannot be parsed!\n {response}")
  

def compare_num_dicts(gen_dict, gt_dict):
  result = {}
  for key in gt_dict:
    try:
      if gt_dict[key] is None or gen_dict[key] is None:
          result[key] = None
      elif key in gen_dict and abs(gen_dict[key] - gt_dict[key]) / max(abs(gt_dict[key]), 1e-10) <= 0.05:
          result[key] = 1
      else:
          result[key] = 0
    except Exception as e:
      print(e)
      #print("gen dict: \n",gen_dict)
      #print("gt dict: \n",gt_dict)
  return result
  # 1 means correct, 0 incorrect, and None is when the GT also doesn't have it


def create_paraphrase_consistency_question(caption_path1, caption2, same_phenom, prompt_save_folder, answer_save_folder):
  """
  caption_path1: filepath of the main anchor caption
  caption2: a string
  same_phenom: a boolean indicating whther the two captions describe the same phenomenon
  """
  
  with open(caption_path1, "r") as file1:
    caption1 = file1.read()
    
  prompt = f"""
  Given the following two time series descriptions, please tell if they describe the same phenomenon.
  
  Answer with "true" if the two describe the same phenomenon, i.e. one can be the paraphrase of the other.
  Answer with "false" if the two describe different phenomena.
  
  Description 1:
  \n
  {caption1}
  \n
  Description 2:
  \n
  {caption2}
  \n
  Return your answer either as "true" or "false", do not explain anything and do not add any text beyond that.
  """
  
  file_path = os.path.join(prompt_save_folder, caption_path1.split('/')[-1])
  with open(file_path, "w") as file:
     file.write(prompt)
     
    
  file_path = os.path.join(answer_save_folder, caption_path1.split('/')[-1])
  with open(file_path, "w") as file:
     file.write(str(same_phenom))

def perturb_caption(caption, model="Google Gemini-2.0-Flash"):
  prompt = f"""Your task is to minimally modify a time series description so that it's meaning is altered. 
    For example, you can switch "increase" with "decrease", "upward" to "downward" a 1 to 2 times, or something more sophisticated. Keep the description structurally identical to the original text, you don't have to alter too much information, altering anywherebetween 1 to 3 parts is enough.
    
    Here's the description to alter:
    \n
    {caption}
    \n
    
    Give your answer in a paragraph of text as the given description, without any explanation and formatting.
  
  """
  response = get_response(prompt=prompt, model=model, temperature=0.6)
  return response

def create_volatility_question(ts_path1, ts2, prompt_save_folder, answer_save_folder):
  ts1 = read_txt_to_num_list(ts_path1)
  prompt = f"""
  Given the following two time series A and B, please identify which one has higher volatility.
  
  A:
  {ts1}
  
  B:
  {ts2}
  
  You must respond only with valid JSON, and no extra text or markdown.
  The JSON schema is:
  {{
    "answer": "A" or "B"
  }}
  Ensure your output parses as JSON with exactly one top-level object containing the answer field.
  
  """
  
  # this function requires that the std information is available in the metadata dictionary. So, checking if it is available must be done before calling this function.
  
  std1 = round(float(np.std(ts1)), 3)
  std2 = round(float(np.std(ts2)), 3)
  
  assert std1 != std2
  
  if std1 > std2:
    answer = "A"
  else:
    answer = "B"
    
    
  file_path = os.path.join(prompt_save_folder, ts_path1.split('/')[-1])
  with open(file_path, "w") as file:
    file.write(prompt)
    
  file_path = os.path.join(answer_save_folder, ts_path1.split('/')[-1])
  with open(file_path, "w") as file:
    file.write(str(answer))
    
def create_mean_question(ts_path1, ts2, prompt_save_folder, answer_save_folder):
  ts1 = read_txt_to_num_list(ts_path1)
  prompt = f"""
  Given the following two time series A and B, please identify which one has higher overall values.
  
  A:
  {ts1}
  
  B:
  {ts2}
  
  You must respond only with valid JSON, and no extra text or markdown.
  The JSON schema is:
  {{
    "answer": "A" or "B"
  }}
  Ensure your output parses as JSON with exactly one top-level object containing the answer field.
  
  """
  
  # this function requires that the std information is available in the metadata dictionary. So, checking if it is available must be done before calling this function.
  mean1 = round(float(sum(ts1)/len(ts1)), 3)
  mean2 = round(float(sum(ts2)/len(ts2)), 3)
  
  assert mean1 != mean2
  
  if mean1 > mean2:
    answer = "A"
  else:
    answer = "B"
    
    
  file_path = os.path.join(prompt_save_folder, ts_path1.split('/')[-1])
  with open(file_path, "w") as file:
    file.write(prompt)
    
  file_path = os.path.join(answer_save_folder, ts_path1.split('/')[-1])
  with open(file_path, "w") as file:
    file.write(str(answer))
          
def create_same_phenomenon_question(ts_path1, ts2, same_phenom, prompt_save_folder, answer_save_folder):
  ts1 = read_txt_to_num_list(ts_path1)
  if same_phenom: # generate ts2 by adding gaussian noise
    alpha = 0.01      # Noise scaling factor (tune as needed)
    gamma = 1.0      # Exponent for controlling nonlinearity
    epsilon = 1e-3   # Small constant to avoid zero std

    mean = 0  # Mean of the Gaussian noise
    ts1 = np.array(ts1)  # ensure it's a NumPy array
    std_dev = alpha * (np.abs(ts1) + epsilon) ** gamma
    noise = np.random.normal(mean, std_dev)
    ts2 = [round(float(val + n), 2) for val, n in zip(ts1, noise)]
  
  if len(ts1) > len(ts2):
    ts1 = ts1[:len(ts2)]
  elif len(ts2) > len(ts1):
    ts2 = ts2[:len(ts1)]
      
  prompt = f"""
  Your task is to determine whether two time series A and B roughly represent the same phenomenon, tolerating noise.
  If they do represent the same pheomenon up to noise, answer with "True", otherwise with "False"
    
  A:
  {ts1}
    
  B:
  {ts2}
    
  You must respond only with valid JSON, and no extra text or markdown.
  The JSON schema is:
  {{
    "answer": "True" or "False"
  }}
  <string> must be an answer string containing only A, B.
  Ensure your output parses as JSON with exactly one top-level object containing the answer field.  
  """

  file_path = os.path.join(prompt_save_folder, ts_path1.split('/')[-1])
  with open(file_path, "w") as file:
    file.write(prompt)
      
  file_path = os.path.join(answer_save_folder, ts_path1.split('/')[-1])
  with open(file_path, "w") as file:
   file.write(str(same_phenom))
    
def create_peak_earlier_question(ts_path1, ts2, prompt_save_folder, answer_save_folder):
  ts1 = read_txt_to_num_list(ts_path1)

  prompt = f"""
  Given two time series A and B, detect which one reaches its maximum earlier.
    
  A:
  {ts1}
    
  B:
  {ts2}
    
  You must respond only with valid JSON, and no extra text or markdown.
  The JSON schema is:
  {{
    "answer": "A" or "B"
  }}
  <string> must be an answer string containing only A, B.
  Ensure your output parses as JSON with exactly one top-level object containing the answer field.  
  """
  
  max_idx_ts1 = ts1.index(max(ts1))
  max_idx_ts2 = ts2.index(max(ts2))

  assert max_idx_ts1 != max_idx_ts2

  if max_idx_ts1 < max_idx_ts2:
    answer = "A"
  else:
    answer = "B"

  file_path = os.path.join(prompt_save_folder, ts_path1.split('/')[-1])
  with open(file_path, "w") as file:
    file.write(prompt)
      
  file_path = os.path.join(answer_save_folder, ts_path1.split('/')[-1])
  with open(file_path, "w") as file:
    file.write(str(answer))
    
def create_bottom_earlier_question(ts_path1, ts2, prompt_save_folder, answer_save_folder):
  ts1 = read_txt_to_num_list(ts_path1)

  prompt = f"""
  Given two time series A and B, detect which one reaches its minimum earlier.
    
  A:
  {ts1}
    
  B:
  {ts2}
    
  You must respond only with valid JSON, and no extra text or markdown.
  The JSON schema is:
  {{
    "answer": "A" or "B"
  }}
  <string> must be an answer string containing only A, B.
  Ensure your output parses as JSON with exactly one top-level object containing the answer field.  
  """
  
  min_idx_ts1 = ts1.index(max(ts1))
  min_idx_ts2 = ts2.index(max(ts2))
  
  assert min_idx_ts1 != min_idx_ts2

  if min_idx_ts1 < min_idx_ts2:
    answer = "A"
  else:
    answer = "B"

  file_path = os.path.join(prompt_save_folder, ts_path1.split('/')[-1])
  with open(file_path, "w") as file:
    file.write(prompt)
      
  file_path = os.path.join(answer_save_folder, ts_path1.split('/')[-1])
  with open(file_path, "w") as file:
    file.write(str(answer))
           
def create_amplitude_question(ts_path1, ts2, prompt_save_folder, answer_save_folder):
  ts1 = read_txt_to_num_list(ts_path1)

  prompt = f"""
  Given two time series A and B, detect which one has a higher amplitude defined as maximum - minimum.
    
  A:
  {ts1}
    
  B:
  {ts2}
    
  You must respond only with valid JSON, and no extra text or markdown.
  The JSON schema is:
  {{
    "answer": "A" or "B"
  }}
  <string> must be an answer string containing only A, B.
  Ensure your output parses as JSON with exactly one top-level object containing the answer field.  
  """
  
  amplitude1 = max(ts1) - min(ts1)
  amplitude2 = max(ts2) - min(ts2)

  assert amplitude1 != amplitude2

  if amplitude1 > amplitude2:
    answer = "A"
  else:
    answer = "B"

  file_path = os.path.join(prompt_save_folder, ts_path1.split('/')[-1])
  with open(file_path, "w") as file:
    file.write(prompt)
      
  file_path = os.path.join(answer_save_folder, ts_path1.split('/')[-1])
  with open(file_path, "w") as file:
    file.write(str(answer))

def perturb_semantically(caption, model="Google Gemini-2.0-Flash"):
  prompt = f"""Your task is to minimally modify a time series description so that it's meaning is altered but the numbers are maintained. 
    For example, you can switch "increase" with "decrease", "upward" to "downward" or something more sophisticated. Keep the description structurally identical to the original text, you don't have to alter too much information, altering anywhere between 1 to 3 parts is enough. Do not edit the numbers.
    
    Here's the description to modify:
    \n
    {caption}
    \n
    
    Give your answer in a paragraph of text as the given description, without any explanation and formatting.
  
  """
  response = get_response(prompt=prompt, model=model, temperature=0.6)
  return response  

def perturb_numerically(caption, model="Google Gemini-2.0-Flash"):
  prompt = f"""Your task is to slightly modify the numbers in a time series description so that its semantics remain the same but the numbers are slightly altered. 
    For example, you can replace "12" with "12.2", "45%" with "46%". Keep the description structurally and semantically identical to the original text; you don't have to alter all numbers but anywhere between 1 to 3 times is enough. Make sure that the altered number still makes sense and fits the scale of the phenomenon.
    
    Here's the description to modify:
    \n
    {caption}
    \n
    
    Give your answer in a paragraph of text as the given description, without any explanation and formatting.
  
  """
  response = get_response(prompt=prompt, model=model, temperature=0.6)
  return response  
    
    
    
def main():
  config = load_config()

  random.seed(config['general']['random_seed'])
  
  """
  caption = "From 2011 to 2018, the agricultural output index in this upper-middle income country shows a consistent upward trend, starting at 91.29 in 2011 and reaching 105.27 in 2018. This indicates a steady growth in agricultural output over these years, with a notable increase of approximately 15% from the beginning to the end of the series. Compared to the historical mean of 53.89, the agricultural output index from 2011 to 2018 is significantly higher, suggesting a period of strong performance relative to the country's longer-term agricultural history. Without global or regional context, it's impossible to determine if this growth is higher, lower, or follows expected patterns."
  
  gen_dict = extract_num_dict_from_text(caption=caption)
  print(gen_dict)
  
  
  with open("/home/ubuntu/thesis/data/samples/new samples no overlap/test/metadata/agriculture_0_test.json", "r") as file:
      metadata = json.load(file)
  
  gt_dict = extract_num_dict_from_dict(metadata=metadata)
  print(gt_dict)
  
  print(compare_num_dicts(gen_dict, gt_dict))
  """
  
  
  """
  gemini_vl_len300 = [
    0.674,    # BERT F1
    0.688,    # BERT Precision
    0.662,    # BERT Recall
    0.6,      # Numeric Score
    0.107,    # BLEU
    0.293,    # ROUGE-L
    0.265,    # METEOR
    0.69,     # Oracle Score
    0.8609    # simCSE
  ]

  gemini_l_len300 = [
      0.698,    # BERT F1
      0.696,    # BERT Precision
      0.7,      # BERT Recall
      0.656,    # Numeric Score
      0.16,     # BLEU
      0.312,    # ROUGE-L
      0.328,    # METEOR
      0.74,     # Oracle Score
      0.8824    # simCSE
  ]
  
  internvl_vl_len300 =[
    0.626,    # BERT F1
    0.619,    # BERT Precision
    0.634,    # BERT Recall
    0.589,    # Numeric Score
    0.122,    # BLEU
    0.235,    # ROUGE-L
    0.29,     # METEOR
    0.376,    # Oracle Score
    0.7226    # simCSE
  ]
  
  internvl_l_len300 = [
    0.632,    # BERT F1
    0.613,    # BERT Precision
    0.653,    # BERT Recall
    0.648,    # Numeric Score
    0.109,    # BLEU
    0.232,    # ROUGE-L
    0.306,    # METEOR
    0.471,    # Oracle Score
    0.7606    # simCSE
  ]
  
  
  gemini_vl_len10 = [
    0.669,    # BERT F1
    0.679,    # BERT Precision
    0.659,    # BERT Recall
    0.733,    # Numeric Score
    0.118,    # BLEU
    0.3,      # ROUGE-L
    0.253,    # METEOR
    0.708,    # Oracle Score
    0.8353    # simCSE
  ]
  
  
  gemini_l_len10 = [
    0.683,    # BERT F1
    0.679,    # BERT Precision
    0.687,    # BERT Recall
    0.754,    # Numeric Score
    0.15,     # BLEU
    0.305,    # ROUGE-L
    0.302,    # METEOR
    0.727,    # Oracle Score
    0.8582    # simCSE
  ]

 
  
  
  internvl_vl_len10 = [
    0.639,    # BERT F1
    0.623,    # BERT Precision
    0.656,    # BERT Recall
    0.681,    # Numeric Score
    0.091,    # BLEU
    0.247,    # ROUGE-L
    0.288,    # METEOR
    0.532,    # Oracle Score
    0.7712    # simCSE
  ]
  
  internvl_l_len10 = [
    0.624,    # BERT F1
    0.62,     # BERT Precision
    0.628,    # BERT Recall
    0.605,    # Numeric Score
    0.084,    # BLEU
    0.24,     # ROUGE-L
    0.248,    # METEOR
    0.432,    # Oracle Score
    0.7183    # simCSE
  ]


  metrics = [
    "BERT F1", "BERT Precision", "BERT Recall", "Numeric",
    "BLEU", "ROUGE-L", "METEOR", "Oracle", "simCSE"
  ]

  # Plot setup
  x = np.arange(len(metrics))  # label locations
  width = 0.35  # width of the bars

  fig, ax = plt.subplots(figsize=(12, 6))
  bars1 = ax.bar(x - width/2, gemini_vl_len300, width, label='Text + Image')
  bars2 = ax.bar(x + width/2, gemini_l_len300, width, label='Text')

  # Labels & formatting
  ax.set_ylabel('Scores')
  ax.set_title('Text vs Text + Image')
  ax.set_xticks(x)
  ax.set_xticklabels(metrics, rotation=45, ha='right')
  ax.legend()

  # Optional: add values on top of bars
  for bar in bars1 + bars2:
      height = bar.get_height()
      ax.annotate(f'{height:.2f}',
                  xy=(bar.get_x() + bar.get_width() / 2, height),
                  xytext=(0, 3),  # vertical offset
                  textcoords="offset points",
                  ha='center', va='bottom', fontsize=8)

  plt.tight_layout()
  plt.grid(True)
  plt.show()
  plt.savefig("/home/ubuntu/thesis/data/samples/len 300/evaluation results/gemini.jpeg")
  """
  
  
  """folder_path = "/home/ubuntu/thesis/data/samples/len 10/time series"

  for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    if filename.endswith(".txt"):
      ts = read_txt_to_num_list(file_path)
      if len(ts) != 300:
        os.remove(file_path)
        print(f"Removed: {file_path}")"""
  
  """nlp = spacy.load("en_core_web_lg")
  stemmer = nltk.PorterStemmer()
  
  gen_text = The Agricultural output index in Upper - middle income from 2011 to 2018 shows a generally upward trend. In 2011, it was 91.29, and by 2018, it reached 105.27. The crop output, animal output, and fish output all contributed to this increase. Compared to global or regional norms, this trend seems to be higher, as the index values are consistently above the base year of 2015 at 100. There are no obvious seasonal patterns in this time series, so it's likely not following typical seasonal trends. Overall, the differences in trends compared to global or regional norms are significant. So, what do you think about this trend? Do you have any other data or thoughts on it?
  
  gt_text = From 2011 to 2018, the agricultural output index in this upper-middle income country shows a consistent upward trend, starting at 91.29 in 2011 and reaching 105.27 in 2018. This indicates a steady growth in agricultural output over these years, with a notable increase of approximately 15% from the beginning to the end of the series. Compared to the historical mean of 53.89, the agricultural output index from 2011 to 2018 is significantly higher, suggesting a period of strong performance relative to the country's longer-term agricultural history. Without global or regional context, it's impossible to determine if this growth is higher, lower, or follows expected patterns.
  
  
  print(extract_numbers_with_semantic_context(gen_text))
  print("\n\n")
  print(extract_numbers_with_semantic_context(gt_text))
  """
  
  
  
  """directory = "/home/ubuntu/thesis/data/samples/new samples no overlap/generated captions/internvl_8b_text"

  for filename in os.listdir(directory):
    if filename.endswith(".txt"):
      old_path = os.path.join(directory, filename)
      new_filename = filename.replace(".txt", "_test.txt")
      new_path = os.path.join(directory, new_filename)
      os.rename(old_path, new_path)

  print("Renaming completed.")"""

  


  """data_ft = [
    [0.655, 0.633, 0.641, 0.625, 0.672, 0.651, 0.642, 0.690, 0.656, 0.633, 0.674, 0.689],
    [0.651, 0.635, 0.644, 0.630, 0.660, 0.651, 0.637, 0.673, 0.654, 0.629, 0.666, 0.677],
    [0.661, 0.632, 0.640, 0.621, 0.685, 0.652, 0.649, 0.709, 0.659, 0.638, 0.682, 0.703],
    [0.594, 0.455, 0.479, 0.589, 0.680, 0.711, 0.588, 0.736, 0.659, 0.326, 0.578, 0.731],
    [0.088, 0.044, 0.071, 0.056, 0.112, 0.114, 0.070, 0.158, 0.065, 0.032, 0.110, 0.136],
    [0.259, 0.210, 0.243, 0.219, 0.296, 0.276, 0.229, 0.305, 0.260, 0.212, 0.291, 0.310],
    [0.282, 0.224, 0.268, 0.252, 0.323, 0.289, 0.265, 0.361, 0.274, 0.217, 0.306, 0.321],
    [0.5684, 0.5342, 0.4446, 0.4103, 0.7283, 0.6090, 0.4726, 0.7176, 0.6560, 0.3532, 0.6614, 0.6656],
    [0.8089, 0.7936, 0.8102, 0.8071, 0.8852, 0.8542, 0.7952, 0.8788, 0.7925, 0.7640, 0.8355, 0.8724],
    [0.655, 0.634, 0.640, 0.625, 0.671, 0.652, 0.642, 0.689, 0.657, 0.632, 0.674, 0.691]
]



  data_pre = [
    [0.637, 0.615, 0.620, 0.608, 0.652, 0.633, 0.627, 0.676, 0.644, 0.615, 0.643, 0.671],
    [0.628, 0.611, 0.616, 0.604, 0.640, 0.628, 0.617, 0.656, 0.637, 0.605, 0.635, 0.659],
    [0.646, 0.619, 0.624, 0.613, 0.664, 0.639, 0.637, 0.697, 0.652, 0.625, 0.653, 0.685],
    [0.551, 0.410, 0.414, 0.488, 0.657, 0.659, 0.547, 0.693, 0.636, 0.282, 0.563, 0.707],
    [0.067, 0.030, 0.051, 0.047, 0.085, 0.082, 0.053, 0.121, 0.063, 0.028, 0.068, 0.107],
    [0.230, 0.184, 0.213, 0.198, 0.263, 0.244, 0.208, 0.278, 0.237, 0.190, 0.244, 0.269],
    [0.259, 0.208, 0.241, 0.240, 0.287, 0.271, 0.249, 0.331, 0.261, 0.206, 0.259, 0.294],
    [0.5224, 0.4096, 0.4081, 0.3929, 0.6919, 0.5638, 0.4375, 0.6327, 0.6208, 0.3747, 0.6101, 0.6043],
    [0.7938, 0.7655, 0.7874, 0.8051, 0.8835, 0.8423, 0.7827, 0.8776, 0.7938, 0.7706, 0.7903, 0.8648]
]


  create_metric_comparisons(data_ft, data_pre, model_name="InternVL", save_path="/home/ubuntu/thesis/source/figs/", label="percentage")"""

  
  
  """for what in ["gt_captions", "metadata", "plots", "time series"]:
    folder_path = f"/home/ubuntu/thesis/data/samples/new samples with overlap/all/{what}"
    train_path = f"/home/ubuntu/thesis/data/samples/new samples no overlap/train/{what}"
    test_path = f"/home/ubuntu/thesis/data/samples/new samples no overlap/test/{what}"
    
    for filename in os.listdir(folder_path):
      if "train" in filename:
        os.remove(os.path.join(folder_path, filename))
        #shutil.copy(os.path.join(folder_path, filename), os.path.join(train_path, filename))
      elif "test" in filename:
        os.remove(os.path.join(folder_path, filename))
        #shutil.copy(os.path.join(folder_path, filename), os.path.join(test_path, filename))
    print(f"\nDone for {what}.")
        """
        
  
  """directory = "/home/ubuntu/thesis/data/samples/test/gt_captions"
  dataset_counts = Counter()

  for filename in os.listdir(directory):
    dataset = filename.split("_")[0]
    dataset_counts[dataset] += 1

  print("Dataset counts:", dataset_counts)"""

  """with open("/home/ubuntu/thesis/data/samples/data_sizes.json", "r") as file:
    data_sizes = json.load(file)

  what = "plots"
  folder_path = f"/home/ubuntu/thesis/data/samples/new/{what}"
  train_path = f"/home/ubuntu/thesis/data/samples/train/{what}"
  test_path = f"/home/ubuntu/thesis/data/samples/test/{what}"
  
  for filename in os.listdir(folder_path):
    if "_" in filename and "." in filename:
      try:
        dataset_name = filename.split("_")[0]
        sample_id = int(filename.split("_")[1].split(".")[0])
        data_size = data_sizes[dataset_name]
        split_idx = int(data_size*0.8)
        
        source_path = os.path.join(folder_path, filename)
        if sample_id < split_idx:
          dest_path = os.path.join(train_path, filename)
        else:
          dest_path = os.path.join(test_path, filename)

        shutil.copy(source_path, dest_path)
      except (IndexError, ValueError):
        print(f"Skipping invalid filename: {filename}")"""

  
  """directory = "/home/ubuntu/thesis/data/samples/test/gt_captions"
  file_count = len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))])
  print(f"Number of files in {directory}: {file_count}")
  
  directory = "/home/ubuntu/thesis/data/samples/test/time series"
  file_count = len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))])
  print(f"Number of files in {directory}: {file_count}")
  
  directory = "/home/ubuntu/thesis/data/samples/test/plots"
  file_count = len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))])
  print(f"Number of files in {directory}: {file_count}")
  
  directory = "/home/ubuntu/thesis/data/samples/test/metadata"
  file_count = len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))])
  print(f"Number of files in {directory}: {file_count}")"""
  
  
  """with open("/home/ubuntu/thesis/data/processed/agricultural_productivity.json", "r") as file:
        data = json.load(file)
  for key in data:
    if 'output_quantity' in data[key]:
      del data[key]['output_quantity']
  with open("/home/ubuntu/thesis/data/processed/agricultural_productivity.json", "w") as file:
      json.dump(data, file, indent=4, separators=(",", ":"))"""
  
  """datasets = [
    "Air Quality", "Border Crossing", "Crime", "Demography", "Road Injuries",
    "Covid", "Co2", "Diet", "Walmart", "Online Retail", "Agriculture"
]

  time_steps = np.array([
      285654436/250, 396618, 38416, 14322, 37354,
      720236, 33842, 233506, 6435, 7416, 48728
  ])

  # Reduce Air Quality weight by half
  adjusted_steps = time_steps.copy()
  adjusted_steps[0] = adjusted_steps[0] / 2

  total_samples = 20000
  min_samples = 500

  # Step 1: Assign minimum samples
  n_datasets = len(datasets)
  assigned = np.full(n_datasets, min_samples)
  remaining = total_samples - min_samples * n_datasets

  # Step 2: Proportional distribution
  weights = adjusted_steps / adjusted_steps.sum()
  proportional = np.floor(weights * remaining).astype(int)

  # Step 3: Final allocation
  final_samples = assigned + proportional

  # Step 4: Fix rounding errors
  while final_samples.sum() < total_samples:
      final_samples[np.argmax(weights)] += 1
  while final_samples.sum() > total_samples:
      final_samples[np.argmax(final_samples)] -= 1

  # Print result
  for name, samples in zip(datasets, final_samples):
      print(f"{name:<16} {samples}")"""
  
  """folder_path="/home/ubuntu/thesis/data/processed"
  timesteps_dict={}
  for filename in os.listdir(folder_path):
     if filename.endswith(".json"):
      file_path = os.path.join(folder_path, filename)
      with open(file_path, "r") as file:
        data = json.load(file)
        def count_numbers(data):
          def is_year(value):
            return isinstance(value, (int, float)) and 1960 <= value <= 2025

          def count_in_list(lst):
            return sum(1 for item in lst if isinstance(item, (int, float)) and not is_year(item))

          def count_in_dict(dct):
            count = 0
            for key, value in dct.items():
              if isinstance(value, list):
                count += count_in_list(value)
              elif isinstance(value, dict):
                count += count_in_dict(value)
            return count

          return count_in_dict(data)

        number_count = count_numbers(data)
        timesteps_dict[filename] = number_count
        print(f"{filename}: {number_count}")
  print(f"{timesteps_dict}")"""
  
  
  """directory = "/home/ubuntu/thesis/data/samples/new samples no overlap/train/time series"

  dataset_names = []
  for filename in os.listdir(directory):
      dataset = filename.split("_")[0]
      if dataset not in dataset_names:
        dataset_names.append(dataset)
  
  print(dataset_names)
  
  avg_lens = {}
  for dataset_name in dataset_names:
    dataset_files = [f for f in os.listdir(directory) if f.startswith(dataset_name)]
    total_length = 0
    count = 0

    for file in dataset_files:
      if file.endswith(".txt"):
        ts = read_txt_to_num_list(os.path.join(directory, file))
        total_length += len(ts)
        count += 1

    if count > 0:
      avg_lens[dataset_name] = total_length / count
    else:
      avg_lens[dataset_name] = 0

  print("Average time series lengths:", avg_lens)"""
     
  
  """
  directory = "/home/ubuntu/thesis/data/samples/captions/generated/qwenvl_vl"

  # Iterate through all files in the directory
  for filename in os.listdir(directory):
    # Extract the numeric ID
    try:
      numeric_id = int(filename.split("_")[1].split(".")[0])
      # Check if the numeric ID is not a multiple of 3
      if numeric_id % 3 != 0:
        # Construct the full file path
        file_path = os.path.join(directory, filename)
        # Remove the file
        os.remove(file_path)
        print(f"Removed: {file_path}")
    except (IndexError, ValueError):
      print(f"Skipping invalid filename: {filename}")"""
  
  """generated_caption = "This time series of daily crime rate in Hollywood starts from 19 in 25 June 2015 and rises to 30 by 30 June 2015. The series then drops down to 9 by the end of July 2015."
  gt_caption = "This time series of daily crime rate in Hollywood starts from 20 in 25 June 2015 and rises to 28 by 30 June 2015. The series then drops down to 9 by the end of July 2015."
  
  #nltk.download('punkt_tab')
  print("\nscore: ", meteor_score(generated_caption, gt_caption))"""
  
  """with open("/home/ubuntu/thesis/data/processed/agricultural_productivity.json", "r") as file:
      json_data = json.load(file)
  metadata, ts = get_sample(dataset_name="agriculture", json_data=json_data)
  print(metadata, ts)
  
  print("\n\n", get_request("agriculture", metadata, ts))"""
  

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

  """gt_caption = "The average daily temperature in San Diego in March 2025 started at 15 degrees Celcius and ended at 17 degrees Celcius. San Diego is a city in California USA and boasts of being one of the cities with the best weather nationwide."
  generated_caption = "The average daily temperature in San Diego in March 2025 started at 15 degrees Celcius and ended at 18 degrees Celcius. San Diego is a city in California USA."
  scores = score_caption(generated_caption, gt_caption, model=config['model']['scoring_model'])
  print(f"Scores: \n{scores}")
"""
  
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


  