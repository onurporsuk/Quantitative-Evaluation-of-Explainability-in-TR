import pickle
import pandas as pd
import torch
from tqdm.notebook import tqdm


def apply_random(files_path, samples, model, tokenizer, file_name, device, only_load=True):
    
    if only_load:
        rand_values = pickle.load(open(files_path + f"{file_name}.pkl", 'rb'))

    else:
        rand_values = []

        with torch.no_grad():
            for sample in tqdm(samples['text']):

                inputs = tokenizer(text, max_length=128, padding='max_length', truncation=True, return_tensors="pt").to(device)
                logits = model(**inputs).logits
                predicted_label = torch.argmax(logits, dim=1).item()

                tokenized_text = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0].tolist())
                rand_values_sample = torch.rand(len(tokenized_text)).tolist()
                rand_values_sample = pd.DataFrame({"Token": tokenized_text, str(predicted_label): rand_values_sample})

                rand_values_sample = rand_values_sample.sort_values(by=str(predicted_label), ascending=False)

                rand_values.append(rand_values_sample)

        pickle.dump(rand_values, open(files_path + f"{file_name}.pkl", 'wb'))
        print(f"File '{file_name}' saved.")

    print(f"'{file_name}' file shape:", rand_values.shape)

    return rand_values