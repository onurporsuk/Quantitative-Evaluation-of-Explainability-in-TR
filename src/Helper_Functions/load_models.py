from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer, AutoConfig


def prepare_models(path_model, device, print_classes=False):
    
    model = AutoModelForSequenceClassification.from_pretrained(path_model).to(device)
    config = AutoConfig.from_pretrained(path_model)
    tokenizer = AutoTokenizer.from_pretrained(path_model, use_fast=True, max_length=128, truncation=True, padding='max_length')
    
    model = model.eval()
    
    pipeline_text = pipeline(task="text-classification", 
                             model=model,
                             tokenizer=tokenizer,
                             max_length=128, truncation=True, padding='max_length',
                             device=device)
    
    # Prepare mappings of different labels
    
    label2id = config.label2id
    id2label = config.id2label

    print(f"\n'{path_model.split('/')[-1]}' is loaded.")
    
    if print_classes:
        for key, value in id2label.items():
            print(f'{key}: {value}')

    return model, tokenizer, config, pipeline_text, label2id, id2label