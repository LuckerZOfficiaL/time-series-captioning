# This is the original version of teacher forcing forward, where the generation is conditioned only on the multimodal representation, no prompt is given.

if teacher_forcing and ground_truth_texts is not None:
    #print("Teacher forcing is active.")

    # Step 1: Tokenize the ground truth prompt (or any custom prompt)
    prompt = "please provide a description to the following time series."
    prompt_input = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
    prompt_input_ids = prompt_input.input_ids  # (B, prompt_len)
    prompt_attention_mask = prompt_input.attention_mask  # (B, prompt_len)

    # Step 2: Convert prompt tokens into embeddings
    prompt_embeddings = self.caption_generator.get_input_embeddings()(prompt_input_ids)  # (B, prompt_len, emb_dim)
    
    # Step 3: Concatenate multimodal embedding `x` with the prompt embeddings
    # Here, `x` is a single token embedding which you prepend to the prompt
    inputs_embeds = torch.cat([x, prompt_embeddings], dim=1)  # (B, prompt_len+1, emb_dim)

    # Step 4: Update attention mask to reflect `x` prepended to the prompt
    combined_attention_mask = torch.cat(
        [torch.ones((batch_size, 1), dtype=torch.long, device=device), prompt_attention_mask], dim=1
    )  # (B, prompt_len+1)

    # Step 5: Tokenize and embed the ground truth texts (for teacher forcing)
    encoded_gt = self.tokenizer(ground_truth_texts, return_tensors="pt", padding=True, truncation=True).to(device)
    gt_input_ids = encoded_gt.input_ids  # (B, seq_len)
    
    # Remove the first token from the ground truth (since it's already used as part of inputs_embeds)
    gt_input_ids = gt_input_ids[:, 1:]  # (B, seq_len-1)

    # Step 6: Pass everything to the GPT decoder
    outputs = self.caption_generator(
        inputs_embeds=inputs_embeds,  # Use the concatenated embeddings
        attention_mask=combined_attention_mask,  # Updated attention mask
        labels=gt_input_ids  # Ground truth labels for teacher forcing
    )

    #print(f"Outputs loss: {outputs.loss}")
    return outputs.loss
