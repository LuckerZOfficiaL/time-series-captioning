from text_encoder import SentenceBERT
from visual_encoder import ViTEncoder
from ts_encoder import TCNEncoder
from fusion_module import LinearFusion
from cross_attention import CrossAttentionWithPrototypes
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import CLIPProcessor, CLIPModel, GPT2LMHeadModel, GPT2Tokenizer
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from helpers import(
    compute_cosine_similarity,
    load_config,
)


class CLIP_Mobtep(torch.nn.Module):
    def __init__(self, prototype_words, tcn_emb_size=64, use_linear_proj=False):
        super(CLIP_Mobtep, self).__init__()
        
        # Load CLIP model
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # TCN encoder for numerical data
        self.ts_encoder = TCNEncoder(embedding_size=tcn_emb_size)  # From scratch

        self.fusion_module = LinearFusion(
            input_size_numeric=tcn_emb_size, input_size_visual=512, input_size_text=512, output_size=768
        )
        self.prototype_attention = CrossAttentionWithPrototypes(prototype_words)
        
        self.use_linear_proj = use_linear_proj
        if self.use_linear_proj:
            self.linear_proj = nn.Linear(768, 768)
            nn.init.xavier_uniform_(self.linear_proj.weight)
            nn.init.zeros_(self.linear_proj.bias)

        self.caption_generator = GPT2LMHeadModel.from_pretrained("gpt2")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        
        self.caption_generator.requires_grad_(False)
        self.clip_model.requires_grad_(False)
    
    def compute_semantic_loss(self, generated_texts, ground_truth_texts):
        """
        Use CLIP's text encoder to ensure generated captions are semantically close to ground-truth.
        """
        # Encode both generated and ground-truth texts
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        generated_input = self.clip_processor(text=generated_texts, return_tensors="pt", padding=True, truncation=True).to(device)
        gt_input = self.clip_processor(text=ground_truth_texts, return_tensors="pt", padding=True, truncation=True).to(device)
        
        # Get embeddings from CLIP (they are still part of the computation graph)
        generated_embeddings = self.clip_model.get_text_features(**generated_input)
        gt_embeddings = self.clip_model.get_text_features(**gt_input)

        # Normalize embeddings before computing similarity
        generated_embeddings = F.normalize(generated_embeddings, dim=-1)
        gt_embeddings = F.normalize(gt_embeddings, dim=-1)

        # Compute cosine similarity and loss
        cosine_sim = F.cosine_similarity(generated_embeddings, gt_embeddings, dim=-1)
        loss = 1 - cosine_sim.mean()  # Contrastive loss (minimize difference)
        
        return loss

    def compute_cross_entropy_loss(self, aligned_embedding, ground_truth_texts):
        """Calculates cross-entropy loss for caption generation."""
        device = aligned_embedding.device

        # Tokenize ground truth captions
        encoded_gt = self.tokenizer(ground_truth_texts, return_tensors="pt", padding=True, truncation=True).to(device)
        input_ids = encoded_gt.input_ids
        attention_mask = encoded_gt.attention_mask

        # Generate logits from GPT-2
        outputs = self.caption_generator(inputs_embeds=aligned_embedding, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss
        return loss
    
    def forward(self, ts_input, text_input, visual_input, ground_truth_texts=None, max_length=250, output_text=False, use_teacher_forcing=False):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Compute multimodal embeddings
        numeric_embedding = self.ts_encoder(ts_input).to(device)
        text_inputs = self.clip_processor(text=text_input, return_tensors="pt", padding=True, truncation=True).to(device)
        text_embedding = self.clip_model.get_text_features(**text_inputs).to(device)
        image_inputs = self.clip_processor(images=visual_input, return_tensors="pt").to(device)
        visual_embedding = self.clip_model.get_image_features(**image_inputs)

        # Fusion + prototype attention
        fused_embedding = self.fusion_module(numeric_embedding, visual_embedding, text_embedding).unsqueeze(1)  
        x = self.prototype_attention(fused_embedding)

        if self.use_linear_proj:
            x = self.linear_proj(x)

        batch_size = x.shape[0]

        if use_teacher_forcing and ground_truth_texts is not None:
            pass

        else:
            """ 🔄 Autoregressive Mode """
            input_ids = torch.full((batch_size, 1), self.tokenizer.bos_token_id, dtype=torch.long, device=device)
            all_logits = []

            for step in range(max_length):
                if step == 0:
                    # First step: Use multimodal input
                    outputs = self.caption_generator(inputs_embeds=x, attention_mask=torch.ones_like(input_ids))
                else:
                    # Later steps: Use predicted tokens
                    outputs = self.caption_generator(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))

                logits = outputs.logits[:, -1, :]
                all_logits.append(logits.unsqueeze(1))

                next_token = torch.argmax(logits, dim=-1, keepdim=True)

                if torch.any(next_token == self.tokenizer.eos_token_id):
                    break

                input_ids = torch.cat([input_ids, next_token], dim=-1)

            logits = torch.cat(all_logits, dim=1)

        if output_text:
            generated_text = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            return generated_text, logits  

        return logits  


    


    def generate_captions(self, aligned_embedding):
        batch_size = aligned_embedding.shape[0]
        
        # Ensure the input tensor has the correct shape (batch_size, sequence_length, embedding_size)
        inputs_embeds = aligned_embedding  # Already has shape (batch_size, 1, 768)

        # Create an attention mask (all ones, since we have a single valid token)
        attention_mask = torch.ones((batch_size, 1), dtype=torch.long, device=aligned_embedding.device)

        # Use GPT-2's generation API to create a description
        outputs = self.caption_generator.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=300,
            num_beams=5,
            no_repeat_ngram_size=2,
            early_stopping=True,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # Decode the output token IDs to text (batch processing)
        generated_descriptions = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        return generated_descriptions



class Mobtep(torch.nn.Module):
    def __init__(self, prototype_words, tcn_emb_size=64, use_linear_proj=False):
        super(Mobtep, self).__init__()
        
        # Pre-trained text and visual encoders, and TCN encoder
        self.text_encoder = SentenceBERT()  # Pretrained
        self.visual_encoder = ViTEncoder()  # Pretrained
        
        self.ts_encoder = TCNEncoder(embedding_size=tcn_emb_size)  # From scratch

        self.fusion_module = LinearFusion(input_size_numeric=tcn_emb_size, input_size_visual=768, input_size_text=768, output_size=768)
        self.prototype_attention = CrossAttentionWithPrototypes(prototype_words) # the unified embedding does cross-attention with text prototypes
        
        self.use_linear_proj = False
        if use_linear_proj:
            self.use_linear_proj = True
            self.linear_proj = nn.Linear(768, 768)
            nn.init.xavier_uniform_(self.linear_proj.weight)
            nn.init.zeros_(self.linear_proj.bias)

        self.caption_generator = GPT2LMHeadModel.from_pretrained("gpt2")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        
        

    def forward(self, ts_input, text_input, visual_input, output_text=False):
        # Encoding each modality
        numeric_embedding = self.ts_encoder(ts_input)
        #print("Numeric embedding: ", numeric_embedding.shape)
        text_embedding = self.text_encoder(text_input)
        #print("Text embedding: ", text_embedding.shape)
        visual_embedding = self.visual_encoder(visual_input)
        #print("Visual embedding: ",visual_embedding.shape)

        # Fusion of embeddings (not shown in this snippet)
        fused_embedding = self.fusion_module(numeric_embedding, visual_embedding, text_embedding)
        #print("Fused: ", fused_embedding.shape)
        
        fused_embedding = fused_embedding.unsqueeze(1)  # Add a new dimension
        #print("Fused and unsqueezed: ", fused_embedding.shape)
        #print(compute_cosine_similarity(fused_embedding))
        

        # Cross-attention with text prototypes
        x = self.prototype_attention(fused_embedding)
        #print("Prototyped: ", x.shape)

        if self.use_linear_proj:
            # Linear projection between embedding and caption generation
            x = self.linear_proj(x)
            #print("Projected: ", x.shape)

        #print(compute_cosine_similarity(x))

        if output_text:
            # Generate description using GPT-2's language model
            captions = self.generate_captions(x)
            return captions

        return x

    def generate_captions(self, aligned_embedding):
        """
        Use GPT-2 to generate a description based on the aligned embedding.
        
        Args:
            aligned_embedding: Tensor of shape (batch_size, 1, embedding_size)
        
        Returns:
            Generated description as a list of strings for the batch.
        """
        batch_size = aligned_embedding.shape[0]
        
        # Ensure the input tensor has the correct shape (batch_size, sequence_length, embedding_size)
        inputs_embeds = aligned_embedding  # Already has shape (batch_size, 1, 768)

        # Create an attention mask (all ones, since we have a single valid token)
        attention_mask = torch.ones((batch_size, 1), dtype=torch.long, device=aligned_embedding.device)

        # Use GPT-2's generation API to create a description
        outputs = self.caption_generator.generate(
            inputs_embeds=inputs_embeds,  # Use embeddings instead of token IDs
            attention_mask=attention_mask,  # Explicitly pass attention mask
            max_new_tokens=300,  # Generate up to 300 new tokens
            num_beams=5,  # Beam search for better quality text
            no_repeat_ngram_size=2,  # Avoid repetition
            early_stopping=True,
            num_return_sequences=1,  # Number of output sequences to return
            pad_token_id=self.tokenizer.eos_token_id,  # Ensure padding behavior is correct
        )

        # Decode the output token IDs to text (batch processing)
        generated_descriptions = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        return generated_descriptions





def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config()


    # Example input data (random for demonstration)
    ts_input = torch.randn(3, 100, 1).to(device)  # Example time series (3 samples, length 100)
    text_input = ["Tell me a story.", "How are you?", "I am Luca."] # these are gonna be metadata with a request
    
    img_paths = ['/home/ubuntu/thesis/data/samples/plots/air quality_0.jpeg', 
                    '/home/ubuntu/thesis/data/samples/plots/crime_0.jpeg', 
                    '/home/ubuntu/thesis/data/samples/plots/demography_0.jpeg']
    images = [Image.open(img_path).convert("RGB") for img_path in img_paths]
    transform = transforms.ToTensor()
    images = [transform(image) for image in images]


    mobtep = CLIP_Mobtep(tcn_emb_size=128, prototype_words=config['mobtep']['anchor_words'], use_linear_proj=False).to(device)
    mobtep.eval()

    #print(mobtep.compute_semantic_loss(["A dog is sleeping", "It's raining"], ['A cat is sleeping', "It is raining"]))

    
    with torch.no_grad():
        output = mobtep(ts_input, text_input, images, output_text=False)

    print(output.shape)  # Expected shape: [3, 1, 768]
    print(output[0])
    #for i, caption in enumerate(output):
    #    print("\n\ni)\n", caption)
    


if __name__ == "__main__":
    main()


