from datasets import load_dataset
from datasets import Dataset
import pickle
import json


def prepare_datasets(path_data, max_length):
    
    ttc4900_eval = Dataset.from_csv(path_data + "eval.csv")
    ttc4900_eval = ttc4900_eval.rename_columns({"labels": "label"})
    ttc4900_eval_512limit = ttc4900_eval.map(lambda sample: {'text': sample['text'][:max_length]})
    
    tr_news = Dataset.from_csv(path_data + "tr_news_df.csv", )
    tr_news = tr_news.rename_columns({old_col: new_col for old_col, new_col in zip(tr_news.column_names, ttc4900_eval.column_names)})
    tr_news = tr_news.select(range(199))
    tr_news_512limit = tr_news.map(lambda sample: {'text': sample['text'][:max_length]})
    
    with open('Data/generated_news.json', 'r', encoding='utf-8') as json_file:
        generated_news = json.load(json_file)
    
    news_texts = [text for text, label in generated_news]
    news_labels = [label for text, label in generated_news]
    
    generated_dataset = Dataset.from_dict({
        "text": news_texts,
        "label": news_labels,
    })

    return ttc4900_eval_512limit, tr_news_512limit, generated_dataset
