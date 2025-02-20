import torch
import torch.nn.functional as F
import pickle
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from general_utils import clear_gpu_memory


def create_predictor_function(model, tokenizer, device):
    
    def predictor(sample):
        
        inputs = tokenizer(sample, max_length=128, padding='max_length', truncation=True, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}

        outputs = model(**inputs)
        tensor_logits = outputs[0]
        
        probas = F.softmax(tensor_logits, dim=1).detach().cpu().numpy()
        
        return probas
    
    return predictor



def apply_lime(files_path, samples, lime_explainer, predictor, model, tokenizer, file_name, num_features=128, only_load=True):
    
    if only_load:
        lime_values, exp_objects = pickle.load(open(files_path + f"{file_name}.pkl", 'rb'))
        
    else:
        lime_values = []
        exp_objects = []

        with torch.no_grad():
            for sample in tqdm(samples['text']):
                
                exp = lime_explainer.explain_instance(sample, predictor, num_features=num_features, num_samples=1000)

                predicted_class_label = np.argmax(np.array(exp.predict_proba))
                lime_values_df = pd.DataFrame(exp.as_list(), columns=['Token', str(predicted_class_label)])
                
                lime_values_df = lime_values_df.sort_values(by=str(predicted_class_label), ascending=False)
                
                lime_values.append(lime_values_df)
                exp_objects.append(exp)

        pickle.dump((lime_values, exp_objects), open(files_path + f"{file_name}.pkl", 'wb'))
        print(f"File '{file_name}' saved.")
                
    print(f"'{file_name}' file length:", len(lime_values))
    
    return lime_values, exp_objects