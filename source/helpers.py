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
import re

random.seed(42)


def get_response(prompt,
                 system_prompt="You are a helpful assistant and you have to generate text on my request.",
                 model="OpenAI GPT-4o",  # "Gemini-1.5-Pro"
                 temperature=0.45,  # Controls randomness (0 = deterministic, 1 = max randomness)
                 top_p=.95,  # Nucleus sampling (0.0 to 1.0, lower = more focused sampling)
                 top_k=40,  # Filters to the top-k highest probability tokens (if supported)
                 max_tokens=300,  # Maximum number of tokens in response
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

        elif model == "Google Gemini-2.0-Flash":
          with open("/home/ubuntu/thesis/.credentials/google", "r") as file:
              google_api_key = file.read().strip()
          client = genai.Client(api_key=google_api_key)

          google_search_tool = Tool(
            google_search = GoogleSearch()
        )

          response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=p,
            config=types.GenerateContentConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
                tools=[google_search_tool],
                response_modalities=["TEXT"],
            )
          )

          text_response = response.text
          #web_metadata = response.candidates[0].grounding_metadata.search_entry_point.rendered_content # To get grounding metadata as web content.
          return text_response

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
                 max_tokens=300,  # Maximum number of tokens in response
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
                    "max_tokens": 256,
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
      series_len = random.randint(5, min(100, 5+int(len(json_data[id][measure])/8)))
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
      series_len = random.randint(5, min(100, 5+int(len(json_data[town]["data"])/8)))
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
      series_len = random.randint(5, min(100, 5+int(len(json_data[port]["data"][means])/8)))
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
      series_len = random.randint(5, min(100, 5+int(len(json_data[patient_id]["data"]["heart rate"])/8)))
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

def get_request(dataset_name, metadata, ts):
  if dataset_name == "air quality":
    request = f"""Here is a time series about {metadata["sampling frequency"]} {metadata["measure"]} in the Indian city of {metadata['city']}: \n {ts} \n Here is the detailed metadata: \n {str(metadata)}.
          \n Describe this time series by focusing on trends and patterns. Discuss concrete numbers you see and pay attention to the dates.
          For numerical values, ensure consistency with the provided time series. If making percentage comparisons, round to the nearest whole number.Report the dates when things happened.
          Use the statistics I provided you for comparing this example to the normalcy.
          Use your broad knowledge of geopolitics, natural events, and economic trends to provide meaningful comparisons.
          Be specific and factual, avoiding broad generalizations.
          Highlight significant spikes, dips, or patterns and explain possible causes based on global or regional factors.
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
          Use your broad knowledge of geopolitics, natural events, and economic trends to provide meaningful comparisons.
          Be specific and factual, avoiding broad generalizations.
          Highlight significant spikes, dips, or patterns and explain possible causes based on global or regional factors.
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
          Use your broad knowledge of geopolitics, natural events, and economic trends to provide meaningful comparisons.
          Be specific and factual, avoiding broad generalizations.
          Highlight significant spikes, dips, or patterns and explain possible causes based on global or regional factors.
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
          Use your broad knowledge of geopolitics, natural events, and economic trends to provide meaningful comparisons.
          Be specific and factual, avoiding broad generalizations.
          Highlight significant spikes, dips, or patterns and explain possible causes based on global or regional factors.
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
          Use your broad knowledge of geopolitics, natural events, and economic trends to provide meaningful comparisons.
          Be specific and factual, avoiding broad generalizations.
          Highlight significant spikes, dips, or patterns and explain possible causes based on global or regional factors.
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

def save_file(data, filepath: str):
    """
    Saves data to a file, supporting strings, lists, dictionaries, and tensors.

    Args:
        data: The data to save.
        filepath (str): The path to the file.
    """
    if isinstance(data, str):
        with open(filepath, 'w') as file:
            file.write(data)
    elif isinstance(data, list):
        with open(filepath, 'w') as file:
            for item in data:
                file.write(str(item) + '\n')
    elif isinstance(data, dict):
        with open(filepath, 'w') as file:
            json.dump(data, file, indent=4, sort_keys=True)
    elif isinstance(data, torch.Tensor):
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
    2. **Replace** them with **concrete facts** by referring to your scientific knowledge and historical events from that period.  
    3. For each fact added, **mention a source, historical reference, or well-documented event**.  
    {"4. If possible, provide URLs to support your statements. If not, ignore this request without commenting." if ask_urls else ""}
    
    **Rules:**  
    - Do NOT modify the original structure of the description beyond factual refinements.  
    - Maintain a natural and fluent writing style.  
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
    You can safely ignore the facts about numbers or numerical comparisons as they are already verified.
    
    Your task is to:  
    1. **Verify all claims or historical references** based on your knowledge.  
    2. **Identify incorrect or unsubstantiated facts** and replace them with accurate ones.  
    3. **Preserve the original writing style and structure**, modifying only incorrect statements.  
    4. **If a fact is unverifiable, state that it is uncertain rather than making assumptions**.  
    
    **Return only the corrected description. Do not add explanations or formatting.**  
    """
    
    response = get_response(prompt=prompt, model=model,
                            temperature=0.3,  # Lower temp for more factual accuracy
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
  
def extract_facts(caption, model="Google Gemini-2.0-Flash"):
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
      
    **Return only the filtered facts without additional output.**  
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

def embed_sentences(sentence_list, model_name="all-MiniLM-L6-v2"):
    """
    Embeds a list of sentences using a pretrained Sentence Transformer model.

    Args:
        sentences (list of str): The list of sentences to embed.
        model_name (str): The name of the Sentence Transformer model to use.

    Returns:
        torch.Tensor: A tensor of shape [N, embedding_size] containing the sentence embeddings.
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(sentence_list, convert_to_tensor=True)
    return embeddings

def save_embeddings_pca(sentence_list, model_name="all-MiniLM-L6-v2"):
    """
    Embeds sentences, performs PCA to reduce dimensionality to 2D, and visualizes them.

    Args:
        sentences (list of str): The list of sentences to embed.
        model_name (str): The name of the Sentence Transformer model to use.
    """
    # 1. Embed Sentences
    model = SentenceTransformer(model_name)
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
    plt.savefig(SAVE_PATH+"/pca.jpeg")
    plt.close()

def augment_prompt_with_facts(prompt: str, all_facts_list: list, all_facts_emb: torch.Tensor, embedding_model, retrieve_k=5) -> str:
    """
    Given a prompt, embed the prompt with embedding model, then find the indices of the top k most similar embeddings from all_facts_emb,
    use these indices to get the actual sentences from all_facts_list, then append these facts row by row to the prompt, resulting in the augmented
    prompt to return.
    """
    prompt_embedding = embedding_model.encode([prompt], convert_to_tensor=True).cpu()
    similarity_scores = cosine_similarity(prompt_embedding, all_facts_emb)[0]

    similarity_tuples = list(enumerate(similarity_scores)) # tuples of (fact_index, similarity_score)
    similarity_tuples.sort(key=lambda x: x[1], reverse=True) # sort by similarity score in descending order

    top_k_indices = [index for index, _ in similarity_tuples[:retrieve_k]]
    top_k_facts = [all_facts_list[index] for index in top_k_indices]

    augmented_prompt = f"{prompt}\n\nHere are some optional facts to consider, if they are helpful:\n"
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

def extract_years(text): # takes a string and returns all the detected years. Years are 4 digits.
  years = re.findall(r'\b\d{4}\b', text)
  years = [int(year) for year in years if int(year) < 2025] # remove non-year numbers and convert to int
  return  years


def split_facts_by_time(facts_list, bin_years=10): # reads through fact_list and categorizes the facts by their time period, storing all in one json file
  min_year = 3000
  max_year = 0

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

  for fact in facts_list:
    years = extract_years(fact)
    for year in years:
      for start_year in time_periods.keys():
        if int(year) >= start_year and int(year) <= start_year + bin_years: # if it's within that bin period
          time_periods[start_year].append(fact)

  return time_periods 



def main():
  prompt = "How was the relative Canadian dollar value compared to USD between 2005 and 2008?"

  embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

  with open("/home/ubuntu/thesis/data/fact bank/all_facts.txt", "r") as file:
    all_facts_list = file.read().splitlines()

  all_facts_emb = torch.load("/home/ubuntu/thesis/data/fact bank/all_facts_emb.pth").cpu()

  augmented_prompt = augment_prompt_with_facts(prompt, all_facts_list, all_facts_emb, embedding_model)
  print(augmented_prompt)


if __name__ == "__main__":
  main()