import torch
from captum.attr import visualization as viz
from tqdm.notebook import tqdm
import pandas as pd
import pickle



def create_model_output_function(model):
    def model_output(input_ids, attention_mask=None):
        return model(input_ids, attention_mask=attention_mask)[0] if attention_mask is not None else model(input_ids)[0]
    return model_output



def construct_input_and_baseline(text, tokenizer, device):
    
    baseline_token_id = tokenizer.pad_token_id
    sep_token_id = tokenizer.sep_token_id 
    cls_token_id = tokenizer.cls_token_id 

    input_ids = tokenizer.encode(text, max_length=128, padding='max_length', truncation=True)
    token_list = tokenizer.convert_ids_to_tokens(input_ids)
  
    # Create baseline input IDs by replacing text IDs with pad token ID
    baseline_input_ids = [cls_token_id if id == cls_token_id else baseline_token_id for id in input_ids]
    baseline_input_ids[-1] = sep_token_id  # Ensure SEP token is at the end

    # Create attention mask
    attention_mask = [1 if id != tokenizer.pad_token_id else 0 for id in input_ids]

    # Trim tokens to remove '##' symbols for better readability
    trimmed_tokens = [token if not token.startswith("##") else token[2:] for token in token_list]

    return torch.tensor([input_ids], device=device), torch.tensor([attention_mask], device=device), torch.tensor([baseline_input_ids], device=device), trimmed_tokens



def interpret_text(text, lig, model, tokenizer, true_class, device):
    
    # Receive input_ids, attention_mask, baseline_input_ids, and tokens from the function
    input_ids, attention_mask, baseline_input_ids, tokens = construct_input_and_baseline(text, tokenizer, device)

    # Pass attention_mask along with input_ids and baseline_input_ids to the model
    attributions, delta = lig.attribute(inputs=input_ids,
                                        baselines=baseline_input_ids,
                                        additional_forward_args=(attention_mask,),
                                        return_convergence_delta=True,
                                        target=true_class)

    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions_sum = attributions / torch.norm(attributions)

    # Get token-level attributions and their corresponding weights
    token_attributions = [(token, attr.item()) for token, attr in zip(tokens, attributions_sum)]

    score_vis = viz.VisualizationDataRecord(
        word_attributions=attributions_sum,
        pred_prob=torch.max(model(input_ids, attention_mask=attention_mask)[0]), 
        pred_class=torch.argmax(model(input_ids, attention_mask=attention_mask)[0]).cpu().numpy(), 
        true_class=true_class,
        attr_class=text,
        attr_score=attributions_sum.sum(),
        raw_input_ids=tokens,
        convergence_score=delta
    )

    return token_attributions, score_vis



def apply_ig(files_path, samples, lig, model, tokenizer, file_name, device, only_load=True):

    if only_load:
        ig_values, visualizations = pickle.load(open(files_path + f"{file_name}.pkl", 'rb'))
        
    else:
        batch_size = 64 
        ig_values, visualizations = [], []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(samples), batch_size)):
                
                batch = samples.select(range(i, min(i + batch_size, len(samples))))
                for sample in batch:
                    
                    token_attributions, score_vis = interpret_text(sample['text'], lig, model, tokenizer, int(sample['label']), device)
                    df = pd.DataFrame(token_attributions, columns=['Token', str(score_vis.pred_class)]).sort_values(by=str(score_vis.pred_class), ascending=False)
                    ig_values.append(df)
                    visualizations.append(score_vis)
        
        pickle.dump((ig_values, visualizations), open(files_path + f"{file_name}.pkl", 'wb'))
        print(f"File '{file_name}' saved.")
    
    print(f"'{file_name}' file length:", len(ig_values))

    return ig_values, visualizations