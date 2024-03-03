import pickle
import pandas as pd
import torch
from tqdm.notebook import tqdm


def apply_prob(files_path, samples, model, tokenizer, file_name, device, only_load=True):
    
    if only_load:
        prob_values = pickle.load(open(files_path + f"{file_name}.pkl", 'rb'))

    else:
        prob_values = []

        with torch.no_grad():
            for sample in tqdm(samples['text']):

                inputs = tokenizer(sample, max_length=128, padding='max_length', truncation=True, return_tensors="pt").to(device)
                outputs = model(**inputs, output_attentions=True)
                logits = outputs.logits
                attentions = outputs.attentions  # Get attention weights
    
                predicted_label = torch.argmax(logits, dim=1).item()

                # Average attention across all heads
                attention = torch.mean(attentions[-1], dim=1)  # Use the last layer's attention
                attention = attention.squeeze(0)  # Remove extra dimension
                
                # Sum the attention weights for each token across the sequence
                token_importance = torch.sum(attention, dim=0)
                # Normalize to get probabilities
                token_probabilities = token_importance / torch.sum(token_importance)

                tokenized_text = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0].tolist())
                
                # Create DataFrame for each sample
                prob_values_sample = pd.DataFrame({"Token": tokenized_text, str(predicted_label): token_probabilities.tolist()})
                prob_values_sample = prob_values_sample.sort_values(by=str(predicted_label), ascending=False)
                
                prob_values.append(prob_values_sample)       

        pickle.dump(prob_values, open(files_path + f"{file_name}.pkl", 'wb'))
        print(f"File '{file_name}' saved.")

    print(f"'{file_name}' file shape:", len(prob_values))

    return prob_values


