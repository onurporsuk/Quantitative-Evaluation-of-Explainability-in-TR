from general_utils import predict, clear_gpu_memory
from tqdm.notebook import tqdm

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



def evaluate_classification(full_text_dataset, parameter_set, label2id):

    y_true = full_text_dataset['label']
    
    full_text_preds = predict(full_text_dataset, **parameter_set)
    y_pred = [label2id[item[0]['label']] for item in full_text_preds]
    
    conf_matrix = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)
    
    metrics = classification_report(y_true, y_pred)
    print("\n\nClassification Report:")
    print(metrics)

    clear_gpu_memory()

    return full_text_preds



def compare_probs(full_text_dataset, full_text_preds, top_tokens, top_k, 
                  model, tokenizer, pipeline=None, pipeline_parameters=None, 
                  id2label=None, device=None):

    results_top_tokens = predict(top_tokens,
                                 model, tokenizer,                            
                                 top_k=top_k,
                                 is_tokenized=True,
                                 mode='custom',
                                 max_length=128,
                                 multi_sample=True,
                                 id2label=id2label,
                                 device=device)

    rows = []
 
    for sample_no, (original_result, top_tokens_result) in enumerate(zip(full_text_preds, results_top_tokens)):
        for item in original_result:

            full_text_label = item['label']
            full_text_score = item['score']
            
            # For top tokens, get the class probability of full text predicted class as the aim is to evaluate the contribution
            matched_tuple = next((t for t in top_tokens_result if t[0] == full_text_label), None)

            top_tokens_label = matched_tuple[0]
            top_tokens_score = matched_tuple[1]

            rows.append({
                'Sample No': sample_no,
                'Actual Label': id2label[full_text_dataset['label'][sample_no]],
                'Pred Label - Full Text': full_text_label,
                'Pred Prob - Full Text': full_text_score,
                'Pred Label - Top Tokens': top_tokens_label,
                'Pred Prob - Top Tokens': top_tokens_score,

                # To check and validate results
                # 'Top Tokens Probas': [(label, round(score, 3)) for label, score in top_tokens_result],
                # 'Top Tokens': top_tokens['text'][sample_no],
                # 'Original Text': full_text_dataset['text'][sample_no]
            })

    return pd.DataFrame(rows)



def evaluate_explanations(results_df, ylim=(-0.005, 0.005)):

    mp_full_text = round(results_df['Pred Prob - Full Text'].mean(), 3)
    mp_top_tokens = round(results_df['Pred Prob - Top Tokens'].mean(), 3)

    print("Mean of Probabilities (MP) of Full Text  :", mp_full_text)
    print("Mean of Probabilities (MP) of Top Tokens :", mp_top_tokens)

    # results_df['Relative Change'] = (results_df['Pred Prob - Top Tokens'] - results_df['Pred Prob - Full Text'])

    # # Calculate the sum and the average of all relative changes
    # total_sum = results_df['Relative Change'].sum()
    # total_ecs = round((total_sum / results_df.shape[0]), 3)
    
    # print(f"Explanation Contribution Score (ECS) for all changes : {total_ecs}")

    # positive_changes = results_df[results_df['Relative Change'] > 0]
    # negative_changes = results_df[results_df['Relative Change'] < 0]
    
    # plt.figure(figsize=(10, 6))
    # plt.bar(positive_changes.index, positive_changes['Relative Change'], color='blue', alpha=0.7, label='Positive Changes')
    # plt.bar(negative_changes.index, negative_changes['Relative Change'], color='red', alpha=0.7, label='Negative Changes')
    # plt.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
    # plt.xlabel('Samples')
    # plt.ylabel('Relative Change (%)')
    # plt.title('Distribution of Relative Changes')
    # plt.ylim(ylim[0], ylim[1])
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    return mp_full_text, mp_top_tokens #, total_ecs



def delete_tokens_incrementally(top_tokens, model, tokenizer, id2label, device, plot=False):

    tokens = top_tokens['text']
    scores = top_tokens['scores']
    token_score_pairs = list(zip(tokens, scores))

    # HF: High first 
    # LF: Low first
    results_df = pd.DataFrame(index=range(len(tokens)), columns=["# of Removed Tokens", 
                                                                 "Label HF", "Proba HF", "Tokens HF", 
                                                                 "Label LF", "Proba LF", "Tokens LF"])
    
    for i in range(len(tokens)):
        
        hf_subset = [pair[0] for pair in token_score_pairs[i:]]
        hf_probas = predict(hf_subset, model, tokenizer, top_k=1, is_tokenized=True, 
                            mode='custom', max_length=128, multi_sample=False, id2label=id2label, device=device)[0][0]
        
        results_df.at[i, "Label HF"] = hf_probas[0]
        results_df.at[i, "Proba HF"] = hf_probas[1]
        results_df.at[i, "Tokens HF"] = hf_subset
        
        lf_subset = [pair[0] for pair in reversed(token_score_pairs[:len(tokens) - i])]
        lf_probas = predict(lf_subset, model, tokenizer, top_k=1, is_tokenized=True, 
                            mode='custom', max_length=128, multi_sample=False, id2label=id2label, device=device)[0][0]
        
        results_df.at[i, "Label LF"] = lf_probas[0]
        results_df.at[i, "Proba LF"] = lf_probas[1]
        results_df.at[i, "Tokens LF"] = lf_subset
    
        results_df.at[i, "# of Removed Tokens"] = i

    if plot:
        display(results_df)
        
        plt.figure(figsize=(10, 6))
        plt.plot(results_df['# of Removed Tokens'], results_df['Proba HF'], marker='o', label='Proba HF')
        plt.plot(results_df['# of Removed Tokens'], results_df['Proba LF'], marker='v', label='Proba LF')
        
        # Annotate label changes
        for i in range(1, len(results_df)):
        
            if results_df['Label HF'].iloc[i] != results_df['Label HF'].iloc[i-1]:
                plt.annotate(f'{results_df["Label HF"].iloc[i]}', 
                             (results_df['# of Removed Tokens'].iloc[i], results_df['Proba HF'].iloc[i]),
                             textcoords="offset points", xytext=(0,10), ha='center')
        
            if results_df['Label LF'].iloc[i] != results_df['Label LF'].iloc[i-1]:
                plt.annotate(f'{results_df["Label LF"].iloc[i]}', 
                             (results_df['# of Removed Tokens'].iloc[i], results_df['Proba LF'].iloc[i]),
                             textcoords="offset points", xytext=(0,10), ha='center')
        
        plt.xlabel('# of Removed Tokens')
        plt.ylabel('Probability')
        plt.legend()
        plt.show()

    return results_df




def aggregate_incremental_metrics(results_list, plot=False, auc=True, log_odds=True, proba=True):

    # The function works for High First
    
    mean_prob_changes = []
    mean_log_odds_changes = []
    aucs = []

    for results_df in results_list:
        
        # Convert probability values to float, and handle any non-numeric issues
        results_df['Proba HF'] = pd.to_numeric(results_df['Proba HF'], errors='coerce')
        results_df = results_df.dropna(subset=['Proba HF'])
        
        initial_proba = results_df['Proba HF'].iloc[0] if not results_df.empty else 0.5

        # Ensure initial probability is within bounds
        initial_proba = np.clip(initial_proba, 1e-6, 1-1e-6)  
        results_df['Proba HF'] = np.clip(results_df['Proba HF'], 1e-6, 1-1e-6)

        # Calculate probability changes and log-odds changes
        proba_changes = results_df['Proba HF'] - initial_proba
        if proba:
            mean_prob_changes.append(proba_changes.mean())

        if log_odds:
            odds = results_df['Proba HF'] / (1 - results_df['Proba HF'])
            initial_odds = initial_proba / (1 - initial_proba)
            log_odds_changes = np.log(odds) - np.log(initial_odds)
            mean_log_odds_changes.append(-log_odds_changes.mean())

        if auc:
            # Calculate AUC using trapezoidal rule
            auc_value = np.trapz(proba_changes, dx=1)
            aucs.append(auc_value)

    if plot:
        plt.figure(figsize=(10, 16)) 
        num_plots = sum([auc, log_odds, proba])
        plot_idx = 1

        if auc:
            plt.subplot(num_plots, 1, plot_idx)
            plt.bar(range(len(aucs)), aucs, color='skyblue')
            plt.xlabel('Sample Index')
            plt.ylabel('AUC Value')
            plt.title('AUC Values Across All Samples')
            plot_idx += 1

        if proba:
            plt.subplot(num_plots, 1, plot_idx)
            plt.bar(range(len(mean_prob_changes)), mean_prob_changes, color='lightgreen')
            plt.xlabel('Sample Index')
            plt.ylabel('Mean Probability Change')
            plt.title('Mean Probability Changes Across All Samples')
            plot_idx += 1

        if log_odds:
            plt.subplot(num_plots, 1, plot_idx)
            plt.bar(range(len(mean_log_odds_changes)), mean_log_odds_changes, color='salmon')
            plt.xlabel('Sample Index')
            plt.ylabel('Mean Log-Odds Change')
            plt.title('Mean Log-Odds Changes Across All Samples')
        
        plt.tight_layout()
        plt.show()

    results = {}
    if auc:
        avg_auc = -np.mean(aucs)
        results['Average AUC'] = [avg_auc]
        print("Average AUC:", avg_auc)

    if proba:
        avg_prob_change = np.mean(mean_prob_changes)
        results['Average Probability Change'] = [avg_prob_change]
        print("Average Probability Change:", avg_prob_change)

    if log_odds:
        avg_log_odds_change = np.mean(mean_log_odds_changes)
        results['Average Log-Odds Change'] = [avg_log_odds_change]
        print("Average Log-Odds Change:", avg_log_odds_change)

    return pd.DataFrame(results)








