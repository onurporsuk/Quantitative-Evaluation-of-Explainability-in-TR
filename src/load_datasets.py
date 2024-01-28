from datasets import load_dataset
from datasets import Dataset
import pickle
import json


def prepare_datasets(path_ttc_4900, path_tr_news, path_interpress, path_tc32):
    
    ttc4900_test = Dataset.from_csv(path_ttc_4900 + "ttc4900_test.csv")
    tr_news_test = Dataset.from_csv(path_tr_news + "tr_news_test_undersampled.csv", )
    interpress_test = Dataset.from_csv(path_interpress + "interpress_test_undersampled.csv", )
    tc32_test = Dataset.from_csv(path_tc32 + "tc32_test_undersampled.csv", )

    return ttc4900_test , tr_news_test, interpress_test, tc32_test