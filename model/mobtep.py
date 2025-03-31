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

        # Freeze pretrained models
        self.caption_generator.requires_grad_(False)
        self.clip_model.requires_grad_(False)
    
    def get_multimodal_embedding(self, ts_input, text_input, visual_input):
        """Extract and fuse embeddings from different modalities"""
        device = next(self.parameters()).device
        
        # Compute multimodal embeddings
        numeric_embedding = self.ts_encoder(ts_input).to(device)
        
        text_inputs = self.clip_processor(text=text_input, return_tensors="pt", padding=True, truncation=True).to(device)
        text_embedding = self.clip_model.get_text_features(**text_inputs).to(device)
        
        image_inputs = self.clip_processor(images=visual_input, return_tensors="pt").to(device)
        visual_embedding = self.clip_model.get_image_features(**image_inputs)

        # Fusion + prototype attention
        fused_embedding = self.fusion_module(numeric_embedding, visual_embedding, text_embedding).unsqueeze(1)  
        embedding = self.prototype_attention(fused_embedding)

        if self.use_linear_proj:
            embedding = self.linear_proj(embedding)
            
        return embedding
    
    def forward(self, ts_input, text_input, visual_input, ground_truth_texts=None, teacher_forcing=False, max_length=250):
        device = next(self.parameters()).device
        
        # Get fused multimodal embedding
        x = self.get_multimodal_embedding(ts_input, text_input, visual_input)
        batch_size = x.shape[0]

        if teacher_forcing and ground_truth_texts is not None:
            # For teacher forcing, we'll use a custom approach to handle the shape mismatch
            
            # Step 1: Get the multimodal context
            mm_embeddings = x  # [batch_size, 1, hidden_size]
            
            # Step 2: Prepare prompt 
            prompt = "please provide a description to the following time series."
            
            # Step 3: Tokenize ground truth texts with prompt prefixed
            prefixed_gt_texts = [f"{prompt} {gt}" for gt in ground_truth_texts]
            encoded_texts = self.tokenizer(prefixed_gt_texts, return_tensors="pt", padding=True, truncation=True).to(device)
            input_ids = encoded_texts.input_ids
            attention_mask = encoded_texts.attention_mask
            
            # Step 4: Create labels - we set the prompt part to -100 (ignored in loss calculation)
            # Find the length of the prompt tokens
            prompt_tokens = self.tokenizer(prompt, return_tensors="pt").to(device).input_ids
            prompt_len = prompt_tokens.size(1)
            
            # Create labels tensor, setting prompt portion to -100
            labels = input_ids.clone()
            labels[:, :prompt_len] = -100
            
            # Step 5: Create position_ids based on attention_mask
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            
            # Step 6: Calculate actual loss using the model
            # First, get token embeddings
            inputs_embeds = self.caption_generator.get_input_embeddings()(input_ids)
            
            # For the first token position in each sequence, replace with multimodal embedding
            inputs_embeds[:, 0, :] = mm_embeddings.squeeze(1)
            
            #print("\ninputs embeds: ", inputs_embeds.shape)
            #print("labels: ", labels.shape)
            #exit()
            # Forward pass through GPT2 with our custom inputs and labels
            outputs = self.caption_generator(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                labels=labels
            )
            
            return outputs.loss
        else:
            # Autoregressive generation
            all_logits = []
            
            # Track which sequences have completed
            completed_sequences = torch.zeros(batch_size, dtype=torch.bool, device=device)
            
            # Step 1: Setup the prompt
            prompt = "please provide a description to the following time series."
            prompt_tokens = self.tokenizer(prompt, return_tensors="pt").to(device).input_ids
            prompt_len = prompt_tokens.size(1)
            
            # Step 2: Initialize the sequence with BOS + prompt tokens
            prefixed_prompt = [f"{self.tokenizer.bos_token} {prompt}" for _ in range(batch_size)]
            encoded_prompts = self.tokenizer(prefixed_prompt, return_tensors="pt", padding=True).to(device)
            input_ids = encoded_prompts.input_ids
            
            # Step 3: Get the embeddings for this sequence
            inputs_embeds = self.caption_generator.get_input_embeddings()(input_ids)
            
            # Replace the first embedding with our multimodal embedding
            inputs_embeds[:, 0, :] = x.squeeze(1)
            
            # Create attention mask and position IDs
            attention_mask = torch.ones_like(input_ids)
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            
            # Step 4: Generate initial sequence
            outputs = self.caption_generator(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids
            )
            
            # Get the predicted token IDs for the next position
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Step 5: Begin autoregressive generation
            # Start with the previously generated prompt + first prediction
            generated_ids = torch.cat([input_ids, next_token], dim=1)
            
            for step in range(1, max_length):
                # Get new outputs based on updated sequence
                outputs = self.caption_generator(
                    input_ids=generated_ids,
                    attention_mask=torch.ones_like(generated_ids)
                )
                
                # Get next token prediction
                next_token_logits = outputs.logits[:, -1, :]
                all_logits.append(next_token_logits.unsqueeze(1))
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Update completed sequences mask
                completed_sequences = completed_sequences | (next_token.squeeze(-1) == self.tokenizer.eos_token_id)
                
                # Append next token to generated sequence
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                
                # Stop if all sequences have completed
                if torch.all(completed_sequences):
                    break
            
            # Combine all logits for return
            if all_logits:
                logits = torch.cat(all_logits, dim=1)
                return logits
            else:
                # Return the logits from the initial generation if no steps were taken
                return outputs.logits
    
    def generate_captions(self, ts_input, text_input, visual_input, prompt_input=None, max_length=250):
        device = next(self.parameters()).device
        
        # Get fused multimodal embedding
        x = self.get_multimodal_embedding(ts_input, text_input, visual_input)
        batch_size = x.shape[0]
        
        # Use default prompt if none provided
        if prompt_input is None:
            prompt = "please provide a description to the following time series."
            prompt_input = [prompt] * batch_size
        elif isinstance(prompt_input, str):
            prompt_input = [prompt_input] * batch_size
        
        # Initialize with BOS + prompt for each sequence
        prefixed_prompts = [f"{self.tokenizer.bos_token} {p}" for p in prompt_input]
        encoded_prompts = self.tokenizer(prefixed_prompts, return_tensors="pt", padding=True).to(device)
        input_ids = encoded_prompts.input_ids
        
        # Get initial embeddings
        inputs_embeds = self.caption_generator.get_input_embeddings()(input_ids)
        
        # Replace first token embedding with multimodal embedding
        inputs_embeds[:, 0, :] = x.squeeze(1)
        
        # Create masks
        attention_mask = torch.ones_like(input_ids)
        
        # Track completed sequences
        completed_sequences = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # Generate initial outputs
        outputs = self.caption_generator(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask
        )
        
        # Get first predicted token
        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        
        # Start with input_ids + first predicted token
        generated_ids = torch.cat([input_ids, next_token], dim=1)
        
        # Autoregressive generation
        for step in range(1, max_length):
            outputs = self.caption_generator(
                input_ids=generated_ids,
                attention_mask=torch.ones_like(generated_ids)
            )
            
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Update completed sequences
            completed_sequences = completed_sequences | (next_token.squeeze(-1) == self.tokenizer.eos_token_id)
            
            # Append to generated sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            
            # Check if all sequences complete
            if torch.all(completed_sequences):
                break
        
        # Find where the prompt ends (to remove from output)
        # Determine prompt length for each example
        prompt_tokens = [self.tokenizer(p, return_tensors="pt").to(device).input_ids[0] for p in prompt_input]
        prompt_lens = [len(p) + 1 for p in prompt_tokens]  # +1 for BOS token
        
        # Extract only the generated part (after prompt) for each sequence
        clean_generations = []
        for i, ids in enumerate(generated_ids):
            # Skip the prefix (BOS + prompt)
            generation = ids[prompt_lens[i]:]
            clean_generations.append(generation)
        
        # Pad to same length for batched decoding
        max_gen_len = max(len(g) for g in clean_generations)
        padded_gens = [torch.cat([g, torch.full((max_gen_len - len(g),), self.tokenizer.pad_token_id, device=device)]) 
                      for g in clean_generations]
        padded_gens = torch.stack(padded_gens)
        
        # Decode
        generated_captions = self.tokenizer.batch_decode(padded_gens, skip_special_tokens=True)
        
        return generated_captions

       


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

    prompt_input = ["Provide a time series description."]*3


    mobtep = CLIP_Mobtep(tcn_emb_size=128, prototype_words=config['mobtep']['anchor_words'], use_linear_proj=False).to(device)
    mobtep.eval()


    with torch.no_grad():
        output = mobtep.generate_captions(ts_input, 
                                        text_input, 
                                        images, 
                                        prompt_input,
                                        max_length=250)

    #with torch.no_grad():
    #  output = mobtep(ts_input, text_input, images, output_text=False, 
    #               ground_truth_texts=['hello world', "university of california", "sapienza university of rome"],
    #               teacher_forcing=False)

    print(output[0])
    #for i, caption in enumerate(output):
    #    print("\n\ni)\n", caption)
    


if __name__ == "__main__":
    main()


