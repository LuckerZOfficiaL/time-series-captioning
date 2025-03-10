import requests 
import random 
import numpy as np 
import pandas as pd
import openai
import json



def get_response(prompt: str,
                 system_prompt="You are a helpful assistant and you have to generate text on my request.",
                 model="GPT-4o-Aug", #"Gemini-1.5-Pro"
                 temperature=0.45,  # Controls randomness (0 = deterministic, 1 = max randomness)
                 top_p=.95,  # Nucleus sampling (0.0 to 1.0, lower = more focused sampling)
                 top_k=40,  # Filters to the top-k highest probability tokens (if supported)
                 max_tokens=150,  # Maximum number of tokens in response,
                 use_openAI = False,
                 ):
    if use_openAI: # if I am using OpenAI's API key
      pass

    else:  
      data = {
          "model": model,
          "messages": [
              {"role": "system", "content": system_prompt},
              {"role": "user", "content": prompt}
          ],
          "temperature": temperature,
          "top_p": top_p,
          "max_tokens": max_tokens,
      }

      API_KEY = "NoBed0fRoses"
      API_ENDPOINT = "https://backend.zzhou.info/v1/chat/completions"

      headers = {
          "Authorization": f"Bearer {API_KEY}",
          "Content-Type": "application/json"
      }


      if top_k is not None:  # Some APIs support top_k, but not all
          data["top_k"] = top_k

      response = requests.post(API_ENDPOINT, headers=headers, json=data)

      if response.status_code == 200:
          result = response.json()
          return result["choices"][0]["message"]["content"]
      else:
          print("Error:", response.status_code, response.text)


def rank_responses(responses_list: list, model="GPT-4o-Aug") -> list: # takes a list of texts, returns a ranking of the indices
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
    measure = random.choice(list(json_data[id].keys())[1:])
    if series_len is None:
      series_len = random.randint(5, min(100, 5+int(len(json_data[id][measure])/8)))
    if start_idx is None:
      start_idx = random.randint(0, len(json_data[id][measure]) - series_len)
    ts = json_data[id][measure][start_idx:start_idx+series_len]
    ts = [round(x, 2) for x in ts]

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
          \n Describe this time series by focusing on trends and patterns. Discuss concrete numbers you see.
          For numerical values, ensure consistency with the provided time series. If making percentage comparisons, round to the nearest whole number.
          Use the statistics I provided you for comparing this example to the normalcy.
          Use your broad knowledge of geopolitics, natural events, and economic trends to provide meaningful comparisons.
          Be specific and factual, avoiding broad generalizations.
          Highlight significant spikes, dips, or patterns and explain possible causes based on global or regional factors.
          You don't have to explicitly report the numeric values of general statistics, you just use them for reference.
          Compare the trends in this time series to global or regional norms, explaining whether they are higher, lower, or follow expected seasonal patterns.
          When making comparisons, clearly state whether differences are minor, moderate, or significant.
          Use varied sentence structures and descriptive language to create engaging, natural-sounding text.
          Avoid repetitive phrasing and overused expressions.

          Answer in a single paragraph of four sentences at most, without bullet points or any formatting.

          """
  elif dataset_name == "crime":
    request = f"""Here is a time series about the number of {metadata["sampling frequency"]} crimes {metadata["town"]}, Los Angeles, from {metadata["start date of the series"]} to {metadata["end date of the series"]}: \n {ts}
          \nThe all-time statistics of {metadata["town"]} until today are: \n Mean: {metadata["general mean in the history of this town"]} \n Standard Deviation: {metadata["general standard deviation in the history of this town"]} \n Minimum: {metadata["general minimum in the history of this town"]} \n Maximum: {metadata["general maximum in the history of this town"]}
          \nAnd the statistics for this specific time series are: \n Mean: {metadata["mean of this specific series"]} \n Standard Deviation: {metadata["standard deviation of this specific series"]} \n Minimum: {metadata["minimum of this specific series"]} \n Maximum: {metadata["maximum of this specific series"]}

         \n Describe this time series by focusing on trends and patterns. Discuss concrete numbers you see.
          For numerical values, ensure consistency with the provided time series. If making percentage comparisons, round to the nearest whole number.
          Use the statistics I provided you for comparing this example to the normalcy.
          Use your broad knowledge of geopolitics, natural events, and economic trends to provide meaningful comparisons.
          Be specific and factual, avoiding broad generalizations.
          Highlight significant spikes, dips, or patterns and explain possible causes based on global or regional factors.
          You don't have to explicitly report the numeric values of general statistics, you just use them for reference.
          Compare the trends in this time series to global or regional norms, explaining whether they are higher, lower, or follow expected seasonal patterns.
          When making comparisons, clearly state whether differences are minor, moderate, or significant.
          Use varied sentence structures and descriptive language to create engaging, natural-sounding text.
          Avoid repetitive phrasing and overused expressions.

          Answer in a single paragraph of four sentences at most, without bullet points or any formatting.

          """

  elif dataset_name == "border crossing":
    request = f"""Here is a time series about the number of {metadata['sampling frequency']} {metadata['means']} crossing the port of {metadata['port']} at the {metadata["border"]} border, starting from {metadata["start date of the series"]}: \n {ts}
          \nThe all-time statistics until today of {metadata['means']} crossing {metadata['port']} are: \n Mean: {metadata["general mean in the history of this port"]} \n Standard Deviation: {metadata["general standard deviation in the history of this port"]} \n Minimum: {metadata["general minimum in the history of this port"]} \n Maximum: {metadata["general maximum in the history of this port"]}
          Note that these all-time statistics are computed from then all the way until today. These are not historical, these are all-time.
          \nThe statistics for this specific time series are: \n Mean: {metadata['mean of this specific series']} \n Standard Deviation: {metadata['standard deviation of this specific series']} \n Minimum: {metadata['minimum in this specific series']} \n Maximum: {metadata['maximum in this specific series']}

           \n Describe this time series by focusing on trends and patterns. Discuss concrete numbers you see.
          For numerical values, ensure consistency with the provided time series. If making percentage comparisons, round to the nearest whole number.
          Use the statistics I provided you for comparing this example to the normalcy.
          Use your broad knowledge of geopolitics, natural events, and economic trends to provide meaningful comparisons.
          Be specific and factual, avoiding broad generalizations.
          Highlight significant spikes, dips, or patterns and explain possible causes based on global or regional factors.
          You don't have to explicitly report the numeric values of general statistics, you just use them for reference.
          Compare the trends in this time series to global or regional norms, explaining whether they are higher, lower, or follow expected seasonal patterns.
          When making comparisons, clearly state whether differences are minor, moderate, or significant.
          Use varied sentence structures and descriptive language to create engaging, natural-sounding text.
          Avoid repetitive phrasing and overused expressions.

          Answer in a single paragraph of four sentences at most, without bullet points or any formatting.
          """

  elif dataset_name == "heart rate":
    request = f"""Here is a time series about the heart rate of a {metadata["category"]}{' ' + metadata["moment"] if "moment" in metadata else ''}, it's measured as instantaneous heart rates across measurements. Here it is: \n {ts}
          \nThe general statistics of this person{' ' + metadata["moment"] if "moment" in metadata else ''} are: \n Mean: {metadata['general mean of this patient in this situation']} \n Standard Deviation: {metadata['general standard deviation of this patient in this situation']} \n Minimum: {metadata['general minimum of this patient in this situation']} \n Maximum: {metadata['general maximum of this patient in this situation']}
          \nThe statistics for this specific time series are: \n Mean: {metadata['mean of this specific series']} \n Standard Deviation: {metadata['standard deviation of this specific series']} \n Minimum: {metadata['minimum of this specific series']} \n Maximum: {metadata['maximum of this specific series']}

          \n Describe this time series by focusing on trends and patterns. Discuss concrete numbers you see.
          For numerical values, ensure consistency with the provided time series. If making percentage comparisons, round to the nearest whole number.
          Use the statistics I provided you for comparing this example to the normalcy.
          Use your broad knowledge of geopolitics, natural events, and economic trends to provide meaningful comparisons.
          Be specific and factual, avoiding broad generalizations.
          Highlight significant spikes, dips, or patterns and explain possible causes based on global or regional factors.
          You don't have to explicitly report the numeric values of general statistics, you just use them for reference.
          Compare the trends in this time series to global or regional norms, explaining whether they are higher, lower, or follow expected seasonal patterns.
          When making comparisons, clearly state whether differences are minor, moderate, or significant.
          Use varied sentence structures and descriptive language to create engaging, natural-sounding text.
          Avoid repetitive phrasing and overused expressions.

          Answer in a single paragraph of four sentences at most, without bullet points or any formatting.
          """

  elif dataset_name == "demography":
    request = f"""I will give you a time series about the {metadata['sampling frequency']} {metadata['attribute']} of {metadata['country']} from {metadata['starting year']} to {metadata['end year']}, it's measured as number per 1000 people.
          {metadata['country']} is categorized as a country with these attributes: {metadata['category by income']}.
           Here is the time series: \n {ts}
          \nHere are the statistics for this specific time series for {metadata['country']}: \n Mean: {metadata['mean of this specific series']} \n Standard Deviation: {metadata['standard deviation of this specific series']} \n Minimum: {metadata['minimum of this specific series']} \n Maximum: {metadata['maximum of this specific series']}
          \nHere is the global average time series for {metadata['attribute']} across all countries in the same period: \n {metadata['global average time series']}, whose standard deviation is {metadata['global standard deviation']}

          \n Describe this time series by focusing on trends and patterns. Discuss concrete numbers you see.
          For numerical values, ensure consistency with the provided time series. If making percentage comparisons, round to the nearest whole number.
          Use the statistics I provided you for comparing this example to the normalcy.
          Use your broad knowledge of geopolitics, natural events, and economic trends to provide meaningful comparisons.
          Be specific and factual, avoiding broad generalizations.
          Highlight significant spikes, dips, or patterns and explain possible causes based on global or regional factors.
          You don't have to explicitly report the numeric values of general statistics, you just use them for reference.
          Compare the trends in this time series to global or regional norms, explaining whether they are higher, lower, or follow expected seasonal patterns.
          When making comparisons, clearly state whether differences are minor, moderate, or significant.
          Use varied sentence structures and descriptive language to create engaging, natural-sounding text.
          Avoid repetitive phrasing and overused expressions.

          Answer in a single paragraph of four sentences at most, without bullet points or any formatting.
          """
  return request


def augment_request(request, n=3, model="GPT-4o-Aug"): # rephrases the request prompt n times and returns the augmentations in a list
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
  else:
    raise ValueError("Unsupported data type")


def add_facts_to_caption(caption, model="Gemini-1.5-Flash-Search"):
  prompt = f"""
   Here is a time series description, read it carefully. 
   \n
   {caption} 
   \n
   The above time series description mentions some vague statements about some social or geopolitical events. Replace these vague statements by referring to your scientific knowledge or explaining concrete events that you know happened in that period of time. Do not affect the rest of the description beyond that. Answer with the refined description in one paragraph, without explaining anything more.
  """
  response = get_response(prompt, model,
                          temperature = 0.7,
                          top_p = 0.85,
            )
  return response