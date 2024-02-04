





def apply_random(samples, tokenizer, threshold):

    # Extract top tokens whose values are above threshold
    
    top_tokens_thresholded = []

    for top_tokens_sample in top_tokens_values:

        label = top_tokens_sample.columns[1]
  
        quantile = top_tokens_sample.iloc[:, 1].quantile(threshold)
        top_tokens_sample = top_tokens_sample[top_tokens_sample.iloc[:, 1] >= quantile]['Token']
   
        # Decode top tokens to generate a full string 
        token_ids = tokenizer.convert_tokens_to_ids(top_tokens_sample)
        decoded_text = tokenizer.decode(token_ids, skip_special_tokens=True)
    
        top_tokens_thresholded.append({
            'text': decoded_text,
            'label': label
        })

    return Dataset.from_list(top_tokens_thresholded)