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


def pad(series, max_len=None, with_value=0):
    padding = max_len - len(series)  # Calculate the padding needed
    if padding > 0:
        padded_s = torch.nn.functional.pad(series, (0, padding), value=with_value)  # Pad the series with `with_value`
    else:
        padded_s = series  # No padding needed if the series is already at max_len
    return padded_s


def bert_score_loss(bert_model, tokenizer, generated_captions, gt_captions):
    """
    Compute the BERT score loss for the generated captions compared to the ground-truth captions.
    The loss is based on minimizing the cosine similarity between the BERT embeddings of the generated
    and ground-truth captions.

    Args:
        bert_model: Pretrained BERT model (should be initialized outside this function).
        generated_captions (list of str): List of generated captions.
        gt_captions (list of str): List of ground-truth captions.

    Returns:
        loss (tensor): The contrastive loss (1 - cosine similarity between generated and ground-truth embeddings).
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

    # Contrastive loss: minimize the cosine distance (maximize similarity)
    loss = 1 - cosine_sim.mean()  # 1 - cosine similarity is the contrastive loss

    return loss


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
        print(f"target {target.shape}, padding {padding.shape}")
        target = torch.cat([target, padding], dim=1)

    # Compute loss, ignoring padding tokens
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1), 
                           ignore_index=pad_token_id, reduction='mean')
    return loss

def main():
    logits = torch.randn((3, 180, 500))
    target = torch.randint(0, 500, (3, 180))
    print(cross_entropy_loss(logits, target, 0))

if __name__ == "__main__":
    main()