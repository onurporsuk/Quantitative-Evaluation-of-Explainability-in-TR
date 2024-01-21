from datasets import load_dataset
from datasets import Dataset
import pickle
import json


def prepare_datasets(path_ttc_4900, path_tr_news, path_interpress, max_length):
    
    ttc4900_test = Dataset.from_csv(path_ttc_4900 + "ttc4900_test.csv")
    ttc4900_test = ttc4900_test.map(lambda sample: {'text': sample['text'][:max_length]})
    
    tr_news_test = Dataset.from_csv(path_tr_news + "tr_news_test_undersampled.csv", )
    tr_news_test = tr_news_test.map(lambda sample: {'text': sample['text'][:max_length]})
    
    # interpress_test = Dataset.from_csv(path_tr_news + "test_set.csv", )
    # # interpress_test_512limit = interpress_test.map(lambda sample: {'text': sample['text'][:max_length]})

    return ttc4900_test , tr_news_test#, interpress_test
