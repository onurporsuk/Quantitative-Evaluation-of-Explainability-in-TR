import pandas as pd
import pickle



def extract_class_shap_values(tokens, weights, num_classes):
    """
    Extracts and sorts SHAP values for a specific class from SHAP values.

    Parameters:
    - shap_values (object): SHAP values object with feature names and values.
    - target_class (int): The specific class for which SHAP values are desired.

    Returns:
    - df_sorted (DataFrame): DataFrame containing 'Token' (feature) and
      the SHAP values for the specified class, sorted in descending order by SHAP values.
    """

    tokens_weights = pd.DataFrame({'Token': tokens})
    
    for i in range(num_classes):
        tokens_weights[i] = weights[:, i]

    weight_sums = tokens_weights.loc[:, tokens_weights.columns != 'Token'].sum()
    
    top_class = weight_sums.idxmax()
    top_weights_sorted = tokens_weights[['Token', top_class]].sort_values(by=top_class, ascending=False)
    
    return top_weights_sorted



def apply_shap(files_path, samples, file_name, shap_explainer, num_classes, only_load=True):

    if only_load:
        shap_values = pickle.load(open(files_path + f"{file_name}.pkl", 'rb'))

    else:
        shap_values = shap_explainer(samples['text'], batch_size=128)

        pickle.dump(shap_values, open(files_path + f"{file_name}.pkl", 'wb'))
        print(f"File '{file_name}' saved.")

    print(f"'{file_name}' file shape:", shap_values.shape)

    extracted_class_tokens = []

    for i, (tokens, weights) in enumerate(zip(shap_values.feature_names, shap_values.values)):

        # Extract and sort SHAP values for the specified class
        df_sorted = extract_class_shap_values(tokens, weights, num_classes)

        # Although tokenizer and pipeline parameters set for max_length 128, SHAP does not apply this
        # Hence, top 128 rows were taken based on the score to handle out-of-max length (i.e., unimportant tokens are removed)
        df_sorted = df_sorted.sort_values(by=df_sorted.columns[1], ascending=False).head(min(128, len(df_sorted)))
        
        extracted_class_tokens.append(df_sorted)
        
    return shap_values, extracted_class_tokens


