import torch
from captum.attr import visualization as viz


def create_model_output_function(model):
    
    def model_output(inputs):
        return model(inputs)[0]
        
    return model_output



def construct_input_and_baseline(text, tokenizer, device):

    max_length = 510
    baseline_token_id = tokenizer.pad_token_id 
    sep_token_id = tokenizer.sep_token_id 
    cls_token_id = tokenizer.cls_token_id 

    text_ids = tokenizer.encode(text, max_length=max_length, truncation=True, add_special_tokens=False)
   
    input_ids = [cls_token_id] + text_ids + [sep_token_id]
    token_list = tokenizer.convert_ids_to_tokens(input_ids)
  
    baseline_input_ids = [cls_token_id] + [baseline_token_id] * len(text_ids) + [sep_token_id]

    # Trim tokens to remove ## symbols
    trimmed_tokens = [token if not token.startswith("##") else token[2:] for token in token_list]

    return torch.tensor([input_ids], device=device), torch.tensor([baseline_input_ids], device=device), trimmed_tokens



def interpret_text(text, lig, model, tokenizer, true_class, device):
    
    input_ids, baseline_input_ids, tokens = construct_input_and_baseline(text, tokenizer, device)
    attributions, delta = lig.attribute(inputs=input_ids,
                                        baselines=baseline_input_ids,
                                        return_convergence_delta=True,
                                        target=true_class)

    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions_sum = attributions / torch.norm(attributions)

    # Get token-level attributions and their corresponding weights
    token_attributions = [(token, attr.item()) for token, attr in zip(tokens, attributions_sum)]

    score_vis = viz.VisualizationDataRecord(
        word_attributions=attributions_sum,
        pred_prob=torch.max(model(input_ids)[0]),
        pred_class=torch.argmax(model(input_ids)[0]).cpu().numpy(),
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
        ig_values = []
        visualizations = []
        
        with torch.no_grad():
            
            for sample in tqdm(samples):
            
                token_attributions, score_vis = interpret_text(sample['text'], lig, model, tokenizer, int(sample['label']), device)
                token_attributions_df = pd.DataFrame(token_attributions, columns=['Token', str(score_vis.pred_class)])

                token_attributions_df = token_attributions_df.sort_values(by=str(score_vis.pred_class), ascending=False)
                
                ig_values.append(token_attributions_df)
                visualizations.append(score_vis)
        
        pickle.dump((ig_values, visualizations), open(files_path + f"{file_name}.pkl", 'wb'))
        print(f"File '{file_name}' saved.")
    
    print(f"'{file_name}' file length:", len(ig_values))

    return ig_values, visualizations