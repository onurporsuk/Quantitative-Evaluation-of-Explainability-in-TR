from transformers import pipeline, BertForSequenceClassification, BertTokenizerFast, AutoConfig


def prepare_models(path_model, device):
    
    model = BertForSequenceClassification.from_pretrained(path_model).to(device)
    tokenizer = BertTokenizerFast.from_pretrained(path_model, use_fast=True)
    config = AutoConfig.from_pretrained(path_model)
    
    model = model.eval()
    
    pipeline_text = pipeline(task="text-classification", 
                             model=model,
                             tokenizer=tokenizer,
                             device=device)
    
    # Prepare mappings of different labels
    
    label2id = config.label2id
    id2label = config.id2label

    print(f"\nLoaded model '{path_model.split('/')[-1]}' has following classes:\n")
    for key, value in id2label.items():
        print(f'{key}: {value}')

    return model, tokenizer, config, pipeline_text, label2id, id2label