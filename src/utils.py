import json
import gc
from collections import namedtuple
from datasets import Dataset

from sklearn.metrics import classification_report, confusion_matrix
import torch
from transformers.pipelines.pt_utils import KeyDataset



def load_hyperparameters(file_path):
    
    with open(file_path, 'r') as file:
        config = json.load(file)
        
    return namedtuple('Config', config.keys())(**config)



def get_model_details(model_config, model_tokenizer):
    
    config_dict = model_config.to_dict()
    
    print("Model Configuration Details:")
    for key, value in config_dict.items():
        print(f"{key}: {value}")

    return None



def predict(text, model, tokenizer, 
            top_k=None, is_tokenized=False, 
            device='cuda', 
            mode='pipeline', text_pipeline=None, pipeline_parameters=None,
            multi_sample=False,
            id2label=None):

    if not isinstance(text, list) and not isinstance(text, Dataset):
        text = [text]

    if mode == 'custom':
        if not is_tokenized:
            inputs = tokenizer(text, 
                               padding='max_length',
                               truncation=True, 
                               return_tensors="pt").to(device)
        else:
            input_ids = torch.tensor([tokenizer.convert_tokens_to_ids(text)])
            inputs = {"input_ids": input_ids.to(device)}

        with torch.no_grad():
            logits = model(**inputs)[0]

        probabilities = torch.softmax(logits, dim=1).squeeze()

        sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)

        if top_k is not None:
            sorted_probs = sorted_probs[:top_k]
            sorted_indices = sorted_indices[:top_k]

        results = []
        for prob, index in zip(sorted_probs.tolist(), sorted_indices.tolist()):
            class_name = id2label[index]
            results.append((class_name, prob))

        return results
        
    elif mode == 'pipeline':

        if multi_sample:
            results = []

            for result in text_pipeline(KeyDataset(text, 'text'),
                                        top_k=top_k,
                                        **pipeline_parameters):
                results.append(result)

        else:
            results = text_pipeline(text,
                                    top_k=top_k,
                                    **pipeline_parameters)
        
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

    return full_text_preds
    


def clear_gpu_memory():

    gc.collect()
    torch.cuda.empty_cache()
    
    return None