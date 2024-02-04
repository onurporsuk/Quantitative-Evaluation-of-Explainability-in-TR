import torch
import torch.nn.functional as F
import pickle
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm


def create_predictor_function(model, tokenizer, device):
    
    def predictor(sample):
        
        inputs = tokenizer(sample, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {key: value.to(device) for key, value in inputs.items()}
    
        outputs = model(**inputs)
        tensor_logits = outputs[0]
        
        probas = F.softmax(tensor_logits, dim=1).detach().cpu().numpy()
        
        return probas
    
    return predictor



def apply_lime(files_path, samples, lime_explainer, predictor, model, tokenizer, file_name, only_load=True):
    
    if only_load:
        lime_values, exp_objects = pickle.load(open(files_path + f"{file_name}.pkl", 'rb'))
        
    else:
        lime_values = []
        exp_objects = []

        with torch.no_grad():
            for sample in tqdm(samples['text']):
                
                # Create a wrapper function for predictor with only the sample
                predictor_wrapper = lambda x, model=model, tokenizer=tokenizer, predictor=predictor: predictor(x)
                
                exp = lime_explainer.explain_instance(sample, predictor_wrapper, num_features=128, num_samples=500)

                predicted_class_label = np.argmax(np.array(exp.predict_proba))
                lime_values_df = pd.DataFrame(exp.as_list(), columns=['Token', str(predicted_class_label)])
                
                lime_values_df = lime_values_df.sort_values(by=str(predicted_class_label), ascending=False)
                
                lime_values.append(lime_values_df)
                exp_objects.append(exp)

        pickle.dump((lime_values, exp_objects), open(files_path + f"{file_name}.pkl", 'wb'))
        print(f"File '{file_name}' saved.")
                
    print(f"'{file_name}' file length:", len(lime_values))
    
    return lime_values, exp_objects