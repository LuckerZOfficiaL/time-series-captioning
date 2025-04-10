import yaml
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
    
# Initialize the BERT model and tokenizer
def initialize_bert_model(model_name="bert-base-uncased"):
    # Load pre-trained BERT model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name).to(device)
    
    # Return both the model and tokenizer
    return model, tokenizer


def compute_cosine_similarity(embeddings):
            # Reshape the input embeddings to [batch_size, embedding_size]
            embeddings = embeddings.view(embeddings.size(0), -1)
            
            # Compute the dot product of the embeddings
            dot_product = torch.matmul(embeddings, embeddings.t())
            
            # Compute the norm of the embeddings
            norm = torch.norm(embeddings, dim=1, keepdim=True)
            
            # Compute the cosine similarity matrix
            cosine_similarity = dot_product / (norm * norm.t())
            
            return cosine_similarity


def load_config(filepath="/home/ubuntu/thesis/source/configs/config.yaml"):
    with open(filepath, "r") as file:
        config = yaml.safe_load(file)
    return config


def pad(tensor, max_len, with_value=0):
    """Pad a 1D tensor to max_len with the given value."""
    if len(tensor) >= max_len:
        return tensor[:max_len]
    else:
        padding = torch.ones(max_len - len(tensor)) * with_value
        return torch.cat([tensor, padding])


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
        score (tensor): average cosine similarity between generated and ground-truth embeddings.
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


def cross_entropy_loss(logits, target, pad_token_id):
    """
    Computes cross-entropy loss, ignoring padding tokens.
    
    logits: (batch_size, seq_len, vocab_size)
    target: (batch_size, seq_len)
    pad_token_id: Token ID used for padding.
    """
    # Shift target sequence for teacher forcing
    target = target[:, 1:]  # Remove <bos> token
    logits = logits[:, :-1, :]  # Align logits with target sequence

    # Handle padding by ensuring target length matches logits length
    target_len = target.size(1)
    logits_len = logits.size(1)

    if target_len < logits_len:
        # Pad target to match logits length
        padding = torch.full((target.size(0), logits_len - target_len), pad_token_id, device=target.device)
        #print(f"target {target.shape}, padding {padding.shape}")
        target = torch.cat([target, padding], dim=1)
    elif target_len > logits_len:
        target = target[:, :logits_len]

    # Compute loss, ignoring padding tokens
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1), 
                           ignore_index=pad_token_id, reduction='mean')
    return loss

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

def main():
    logits = torch.randn((3, 180, 500))
    target = torch.randint(0, 500, (3, 180))
    print(cross_entropy_loss(logits, target, 0))

if __name__ == "__main__":
    main()