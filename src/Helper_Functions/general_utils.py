import json
import gc
from collections import namedtuple
from datasets import Dataset
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
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


def predict(text, model, tokenizer, 
            top_k=None, is_tokenized=True, 
            mode='pipeline', text_pipeline=None, pipeline_parameters=None,
            max_length=128,
            multi_sample=True,
            id2label=None,
            device=None):

    if not isinstance(text, list) and not isinstance(text, Dataset):
        text = [text]

    if mode == 'custom' and is_tokenized and multi_sample:
        results = []

        for sample in text:
          
            input_ids = torch.tensor([tokenizer.convert_tokens_to_ids(sample['text'])])
            inputs = {"input_ids": input_ids.to(device)}
          
            with torch.no_grad():
                logits = model(**inputs)[0]
    
            probabilities = torch.softmax(logits, dim=1).squeeze()
    
            sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)
    
            if top_k is not None:
                sorted_probs = sorted_probs[:top_k]
                sorted_indices = sorted_indices[:top_k]
    
            sample_result = []
            for prob, index in zip(sorted_probs.tolist(), sorted_indices.tolist()):
                class_name = id2label[index]
                sample_result.append((class_name, prob))

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


def evaluate_classification(full_text_dataset, parameter_set, label2id):

    y_true = full_text_dataset['label']
    
    full_text_preds = predict(full_text_dataset, **parameter_set)
    y_pred = [label2id[item[0]['label']] for item in full_text_preds]
    
    conf_matrix = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)
    
    metrics = classification_report(y_true, y_pred)
    print("\n\nClassification Report:")
    print(metrics)

    clear_gpu_memory()

    return full_text_preds


def apply_thresholding(top_tokens_samples, threshold):

    # Extract top tokens whose values are above threshold
    
    top_tokens_thresholded = []

    for top_tokens_sample in top_tokens_samples:

        label = top_tokens_sample.columns[1]
  
        quantile = top_tokens_sample.iloc[:, 1].quantile(threshold)
        top_tokens_sample = top_tokens_sample[top_tokens_sample.iloc[:, 1] >= quantile]['Token']
        
        top_tokens_thresholded.append({
            'text': top_tokens_sample,
            'label': label
        })

    return Dataset.from_list(top_tokens_thresholded)


def compare_probs(full_text_dataset, full_text_preds, top_tokens, top_k, 
                  model, tokenizer, pipeline=None, pipeline_parameters=None, 
                  id2label=None, device=None):

    results_top_tokens = predict(top_tokens,
                                 model, tokenizer,                            
                                 top_k=top_k,
                                 is_tokenized=True,
                                 mode='custom',
                                 max_length=128,
                                 multi_sample=True,
                                 id2label=id2label,
                                 device=device)

    rows = []
 
    for sample_no, (original_result, top_tokens_result) in enumerate(zip(full_text_preds, results_top_tokens)):
        for item in original_result:

            full_text_label = item['label']
            full_text_score = item['score']
            
            # For top tokens, get the class probability of full text predicted class as the aim is to evaluate the contribution
            matched_tuple = next((t for t in top_tokens_result if t[0] == full_text_label), None)

            top_tokens_label = matched_tuple[0]
            top_tokens_score = matched_tuple[1]

            rows.append({
                'Sample No': sample_no,
                'Actual Label': id2label[full_text_dataset['label'][sample_no]],
                'Pred Label - Full Text': full_text_label,
                'Pred Prob - Full Text': full_text_score,
                # 'Pred Label - Top Tokens': top_tokens_label,
                'Pred Prob - Top Tokens': top_tokens_score,
                'Top Tokens Probas': [(label, round(score, 3)) for label, score in top_tokens_result],
                'Top Tokens': top_tokens['text'][sample_no],
                'Original Text': full_text_dataset['text'][sample_no]
            })

    return pd.DataFrame(rows)


def evaluate_explanations(results_df, ylim=(-0.005, 0.005)):

    classification_acc = accuracy_score(results_df['Actual Label'], results_df['Pred Label - Full Text'])
    print("\nClassification accuracy                             : ", round(classification_acc, 3))

    ecs_full_text = round(results_df['Pred Prob - Full Text'].mean(), 3)
    ecs_top_tokens = round(results_df['Pred Prob - Top Tokens'].mean(), 3)

    print("Explanations Contribution Score (ECS) of Full Text  : ", ecs_full_text)
    print("Explanations Contribution Score (ECS) of Top Tokens : ", ecs_top_tokens)

    results_df['Relative Change'] = (results_df['Pred Prob - Top Tokens'] - results_df['Pred Prob - Full Text'])

    # Filter samples with positive change
    positive_changes = results_df[results_df['Relative Change'] > 0]
    positive_orc = round(positive_changes['Relative Change'].mean() * 100, 3)
    
    # Filter samples with negative change
    negative_changes = results_df[results_df['Relative Change'] < 0]
    negative_orc = round(negative_changes['Relative Change'].mean() * 100, 3)
    
    print(f"Overall Relative Change (ORC) positive changes      :  {positive_orc} %")
    print(f"Overall Relative Change (ORC) negative changes      : {negative_orc} %")

    plt.figure(figsize=(10, 6))
    plt.bar(positive_changes.index, positive_changes['Relative Change'], color='blue', alpha=0.7, label='Positive Changes')
    plt.bar(negative_changes.index, negative_changes['Relative Change'], color='red', alpha=0.7, label='Negative Changes')
    plt.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
    plt.xlabel('Samples')
    plt.ylabel('Relative Change (%)')
    plt.title('Distribution of Relative Changes')
    plt.ylim(ylim[0], ylim[1])
    plt.legend()
    plt.tight_layout()
    plt.show()

    return classification_acc, ecs_full_text, ecs_top_tokens, positive_orc, negative_orc


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
    

def clear_gpu_memory():

    gc.collect()
    torch.cuda.empty_cache()
    
    return None