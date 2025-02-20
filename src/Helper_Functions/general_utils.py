import json
import gc
from collections import namedtuple
from datasets import Dataset
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

import torch
from transformers.pipelines.pt_utils import KeyDataset



def load_hyperparameters(file_path):
    
    with open(file_path, 'r') as file:
        config = json.load(file)
        
    return config



def get_model_details(model_config, model_tokenizer):
    
    config_dict = model_config.to_dict()
    
    print("Model Configuration Details:")
    for key, value in config_dict.items():
        print(f"{key}: {value}")

    return None



def analyze_dataset(dataset, figsize_bar, figsize_char, figsize_word, ylim_char, max_val_pos_char, ylim_word, max_val_pos_word, color, name, rotation=0):

    print(f"\nAnalysis of {name} Dataset\n")
    
    # Sample Distribution
    
    plt.figure(figsize=figsize_bar) 
    dataset['label'].value_counts().plot(kind='bar', color=color)
    plt.xlabel('Categories')
    plt.ylabel('Count')
    plt.xticks(rotation=rotation)
    plt.tight_layout()
    plt.show()

    # Character Statistics
    
    character_stats = dataset['text'].str.len()

    print("\nStatistical measures for input text (character-level):\n")
    display(pd.DataFrame(character_stats).describe().round(2))
    
    plt.figure(figsize=figsize_char)
    plt.bar(character_stats.index, character_stats.values, color=color)
    plt.xlabel('Number of Documents')
    plt.ylabel('Frequency')
    plt.title('Distribution of Number of Characters in Input Text')
    plt.ylim(0, character_stats.max() * ylim_char)
    # max_value = character_stats.max()
    # plt.text(
    #     character_stats.idxmax(),
    #     max_value * max_val_pos_char, 
    #     f'Max: {max_value}',  
    #     ha='center',
    #     va='bottom'
    # )
    plt.show()

    # Word Statistics

    word_stats = dataset['text'].str.split().apply(len)
    word_stats.to_excel(f"word_stats_{name}.xlsx")  

    print("Statistical measures for input text (word-level):\n")
    display(pd.DataFrame(word_stats).describe().round(2))

    plt.figure(figsize=figsize_word)
    plt.bar(word_stats.index, word_stats.values, color=color)
    plt.xlabel('Number of Documents')
    plt.ylabel('Frequency')
    plt.title('Distribution of Number of Words in Input Text')
    plt.ylim(0, word_stats.max() * ylim_word) 
    # max_word_value = word_stats.max()
    # plt.text(
    #     word_stats.idxmax(),
    #     max_word_value * max_val_pos_word,
    #     f'Max: {max_word_value}',
    #     ha='center',
    #     va='bottom'
    # )
    plt.show()

    return None



def predict(text, model, tokenizer, 
            top_k=None, is_tokenized=True, 
            mode='pipeline', text_pipeline=None, pipeline_parameters=None,
            max_length=128,
            multi_sample=True,
            id2label=None,
            device=None):

    if not isinstance(text, list) and not isinstance(text, Dataset):
        text = [text]

    if mode == 'custom' and is_tokenized:

        if multi_sample is False:
            text_samples = [text]
        else:
            text_samples = [entry['text'] for entry in text]

        # Convert tokens to IDs and calculate the maximum sequence length for padding
        input_ids = [torch.tensor(tokenizer.convert_tokens_to_ids(sample), dtype=torch.long) for sample in text_samples]
        max_length = max(len(ids) for ids in input_ids)
        
        # Pad sequences and create attention masks
        input_ids_padded, attention_masks = [], []
        for ids in input_ids:
            
            padded_ids = torch.cat([ids, torch.tensor([tokenizer.pad_token_id] * (max_length - len(ids)), dtype=torch.long)])
            input_ids_padded.append(padded_ids)
            attention_masks.append(torch.cat([torch.ones(len(ids)), torch.zeros(max_length - len(ids))]))
        
        # Stack the input IDs and attention masks
        input_ids_tensor = torch.stack(input_ids_padded).to(device)
        attention_masks_tensor = torch.stack(attention_masks).to(device)
        
        batch_size = 128
        results = []
        
        for i in range(0, input_ids_tensor.size(0), batch_size):
            batch_inputs = {
                "input_ids": input_ids_tensor[i:i + batch_size],
                "attention_mask": attention_masks_tensor[i:i + batch_size]
            }
        
            with torch.no_grad():
                logits = model(**batch_inputs).logits
        
            probabilities = torch.softmax(logits, dim=1)
            sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)
        
            for prob, indices in zip(sorted_probs, sorted_indices):
                sample_result = [(id2label[idx.item()], prob.item()) for idx, prob in zip(indices, prob)]
                results.append(sample_result)
            
        return results
        
    elif mode == 'pipeline':

        if multi_sample:
            results = []

            for result in text_pipeline(KeyDataset(text, 'text'), top_k=top_k, **pipeline_parameters, max_length=max_length):
                results.append(result)
            
        else:
            results = text_pipeline(text, top_k=top_k, **pipeline_parameters, max_length=max_length)
        
        return results if multi_sample else results[0]
        
    else:
        raise ValueError("Invalid mode. Please choose either 'custom' or 'pipeline'.")



def apply_thresholding(top_tokens_samples, top_fraction):
    
    top_tokens_thresholded = []

    for top_tokens_sample in top_tokens_samples:

        num_elements = int(len(top_tokens_sample) * (1-top_fraction))
        
        top_elements = top_tokens_sample.nlargest(num_elements, top_tokens_sample.columns[1])

        top_tokens_thresholded.append({
            'text': top_elements['Token'],
            'scores': top_elements.iloc[:, 1],
            'label': top_tokens_sample.columns[1]
        })

    return Dataset.from_list(top_tokens_thresholded)



def mask_below_threshold(top_tokens_samples, threshold):
    tokens_masked = []

    for top_tokens_sample in top_tokens_samples:

        quantile = top_tokens_sample.iloc[:, 1].quantile(threshold)
        label = top_tokens_sample.columns[1]
        
        # Apply the mask to tokens below the threshold and collect scores
        top_tokens_sample['Token'] = top_tokens_sample.apply(
            lambda row: '[MASKED]' if row[label] < quantile else row['Token'], axis=1
        )
        
        tokens_masked.append({
            'text': top_tokens_sample['Token'],
            'scores': top_tokens_sample[label],
            'label': label
        })

    return Dataset.from_list(tokens_masked)



def clear_gpu_memory():

    gc.collect()
    torch.cuda.empty_cache()
    
    return None