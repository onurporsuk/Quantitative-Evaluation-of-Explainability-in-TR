import pandas as pd
import pickle


def extract_weights(features, values):

    '''
    Extracts weights associated with tokens across classes.

    Args:
    - features (list): A list of feature names or tokens.
    - values (numpy.ndarray): A 2D array containing weights associated with features 
      across different classes.

    Returns:
    - features_values (pandas.DataFrame): A DataFrame containing feature names in the 
      'Tokens' column and their respective weights for each class across columns.

    Example:
    >>> features = ['feature_1', 'feature_2', 'feature_3']
    >>> weights = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    >>> extracted = extract_weights(features, weights)
    >>> print(extracted)
         Tokens    0    1    2
    0  feature_1  0.1  0.2  0.3
    1  feature_2  0.4  0.5  0.6
    2  feature_3  0.0  0.0  0.0
    '''
    
    features_values = pd.DataFrame({'Token': features})
    for i in range(7):
        features_values[i] = values[:, i]

    return features_values



def extract_top_weights(tokens_weights):

    '''
    Extracts the top weights of the top output associated with tokens.

    Args:
    - tokens_weights (pandas.DataFrame): A DataFrame containing tokens in the 'Tokens' 
      column and their respective weights for each class across columns.

    Returns:
    - top_weights_sorted (pandas.DataFrame): A DataFrame sorted by the weights of the 
      top class, showing tokens and their weights for the top class.

    Example:
    >>> tokens_weights = pd.DataFrame({
    ...     'Tokens': ['token_1', 'token_2', 'token_3'],
    ...     0: [0.1, 0.2, 0.3],
    ...     1: [0.4, 0.5, 0.6],
    ...     2: [0.7, 0.8, 0.9]
    ... })
    >>> extracted_top = extract_top_weights(tokens_weights)
    >>> print(extracted_top)
        Tokens    2
    2  token_3  0.9
    1  token_2  0.6
    0  token_1  0.3
    '''

    weight_sums = tokens_weights.loc[:, tokens_weights.columns != 'Token'].sum()
    
    top_class = weight_sums.idxmax()
    top_weights_sorted = tokens_weights[['Token', top_class]].sort_values(by=top_class, ascending=False)
    
    return top_weights_sorted



def extract_top_tokens_weights(shap_values):

    '''
    Extracts top tokens and their corresponding weights from SHAP values.
    
    Args:
    - shap_values: SHAP values object
    
    Returns:
    - list_all: List of dictionaries containing tokens and their weights for all features
    - list_top: List of dictionaries containing top tokens and their weights for each feature
    '''

    list_all = []
    list_top = []
    
    for index, (tokens, weights) in enumerate(zip(shap_values.feature_names, shap_values.values)):
    
        list_all.append(extract_weights(tokens, weights))
        list_top.append(extract_top_weights(list_all[index]))

    return list_all, list_top



def apply_shap(files_path, samples, file_name, shap_explainer, only_load=True):

    if only_load:
        shap_values = pickle.load(open(files_path + f"{file_name}.pkl", 'rb'))

    else:
        shap_values = shap_explainer(samples['text'])
        pickle.dump(shap_values, open(files_path + f"{file_name}.pkl", 'wb'))
        print(f"File '{file_name}' saved.")

    print(f"'{file_name}' file shape:", shap_values.shape)
    
    return shap_values