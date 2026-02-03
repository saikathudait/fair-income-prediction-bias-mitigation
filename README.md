# Mitigating Bias in Machine Learning Models  
### Fairness Evaluation and Reweighing for Income Prediction

## Overview
This project focuses on detecting and mitigating bias in a machine learning model used to predict whether an individual earns more than $50K annually. The analysis highlights fairness concerns related to sensitive attributes such as gender and race and applies a fairness-aware reweighing technique to reduce discriminatory outcomes.

## Dataset
- **Name:** Adult Census Income Dataset
- **Source:** UCI Machine Learning Repository
- **Target Variable:** Income (>50K or <=50K)
- **Protected Attributes:** Sex, Race
- **File:** `adult.csv`

## Tools and Libraries
- Python  
- Pandas, NumPy  
- Scikit-learn  
- AIF360 (IBM Fairness Toolkit)  
- Matplotlib, Seaborn  

## Methodology
1. Data cleaning and preprocessing  
2. Label encoding of categorical variables  
3. Training a Random Forest Classifier  
4. Evaluating model accuracy and bias  
5. Measuring fairness using:
   - Demographic Parity Difference  
   - Average Odds Difference  
6. Applying Reweighing to mitigate bias  
7. Re-evaluating accuracy and fairness metrics  

## Key Results
- Baseline model achieved strong predictive accuracy
- Fairness analysis revealed bias against women and non-white groups
- Reweighing reduced bias while maintaining similar accuracy
- Demonstrates a practical balance between fairness and performance

## Files Included
- `fair-income-prediction-bias-mitigation.ipynb` – Main Python implementation  
- `adult.csv` – Dataset used for training and evaluation  
- `README.md` – Project documentation  

## Ethical Significance
This project demonstrates how fairness-aware machine learning techniques can improve ethical decision-making in sensitive domains such as hiring, promotions, and income prediction.

## How to Run
```bash
pip install pandas numpy scikit-learn aif360 matplotlib seaborn
python fair-income-prediction-bias-mitigation.ipynb

