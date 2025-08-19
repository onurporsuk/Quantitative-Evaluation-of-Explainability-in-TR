# Evaluation of Local Explainability Methods in Turkish Text Classification Tasks

## Overview
This repository contains the research article **"Evaluation of Local Explainability Methods in Turkish Text Classification Tasks"**, which explores and evaluates explainability techniques applied to transformer-based models for the Turkish language.

### Abstract
Complex transformer models are widely used for text classification tasks but often function as "black boxes," making their decision-making process difficult to interpret. This study evaluates local explainability techniques—**SHAP, LIME, and Integrated Gradients (IG)**—on **BERT-based models (BERTurk and TurkishBERTweet)** for Turkish text classification. We introduce **quantitative evaluation metrics** such as **Mean of Probabilities (MP) and Incremental Deletion** to compare the effectiveness of these methods in preserving model prediction probabilities. Our results demonstrate that **IG is the most effective technique**, particularly in capturing key tokens in morphologically rich Turkish text. We discuss challenges such as **tokenization issues, computational costs, and the lack of ground-truth explanations**, providing insights into language-specific explainability research.

## Key Contributions
- Evaluation of explainability methods on **Turkish text classification** using transformer models.
- **Comparison of SHAP, LIME, and IG** on **BERTurk and TurkishBERTweet**.
- **Introduction of novel evaluation approaches**:  
  - **Mean of Probabilities (MP)** for quantifying token importance.  
  - **Incremental Deletion** to assess how removing key tokens impacts predictions.
- **Findings**: IG consistently outperforms SHAP and LIME, proving to be a **robust and efficient explainability method for Turkish NLP**.
- **Discussion of key challenges** in explainability for morphologically rich languages.

## Article Contents
The article includes:
- **Introduction**: Background on explainability in NLP and challenges specific to Turkish text.
- **Methodology**: Explanation of models, datasets, and explainability techniques.
- **Experiments**: Evaluation results and performance comparison.
- **Discussion**: Insights, challenges, and limitations of explainability methods in Turkish.
- **Conclusion & Future Work**: Summary of findings and directions for future research.

## Citation

If you use this work, please cite as:

Porsuk, O., Yıldırım, S., & Başar, A. (2025). Evaluation of Local Explainability Methods in Turkish Text Classification Tasks. In K. Arai (Ed.), *Intelligent Computing. CompCom 2025. Lecture Notes in Networks and Systems* (Vol. 1423). Springer, Cham. https://doi.org/10.1007/978-3-031-92602-0_30
