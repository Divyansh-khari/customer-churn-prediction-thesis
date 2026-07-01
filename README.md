# Customer Churn Prediction Through Comparative Machine Learning Analysis

MSc Data Science | Final Thesis | Liverpool John Moores University (LJMU)

**Author:** Divyansh Khari  
**Supervisor:** Dr. Rajesh Kulkarni  
**Submitted:** July 2026

---

## Overview

This repository contains the complete analysis notebook supporting my MSc dissertation on customer churn prediction. The work compares eight machine learning classifiers across three tiers of complexity — classical, ensemble, and deep learning — on a naturally balanced dataset of 440,832 enterprise customer records.

The study addresses four research questions:

1. Which model family most accurately predicts customer churn on a large naturally balanced dataset?
2. Do ensemble and deep-learning models outperform classical approaches, and why?
3. Which customer attributes are the most influential predictors of churn?
4. What is the estimated annual financial benefit of deploying the best-performing model?

---

## Key findings

- **Best model:** Single-layer MLP Classifier — 96.24% accuracy, ROC-AUC of 0.9906, F1 score 96.63%
- **Convergence finding:** A 1,901-parameter single-layer MLP matches an 8,151-parameter three-layer alternative to within 0.0005 cross-validation accuracy — supporting the position that well-regularised simple networks match deeper architectures on tabular data
- **Cross-framework reproduction:** The same performance ceiling is reached independently in Scikit-learn (MLPClassifier) and Keras/TensorFlow (Sequential ANN), confirming the result is a property of the data rather than any specific implementation
- **Feature drivers:** Four features dominate predictive importance across three independent models (MLP, Random Forest, and Gradient Boosting) and features include Average Spend per Tenure, Support Call Rate, Contract Length (Monthly), and Payment Delay Rate"
- **Business impact:** Baseline annual saving of $4.73 million, robust across cost-sensitivity variations from $2.34M (conservative) to over $7M (favourable)

---

## Methodology

The analysis follows the CRISP-DM framework, with all eight classifiers evaluated under uniform conditions:

- **Dataset:** 440,832 anonymised customer records (naturally balanced: 56.7% churn / 43.3% retention)
- **Data split:** 80:20 stratified train-test split (352,665 training / 88,167 held-out test)
- **Model selection:** 5-fold cross-validation
- **Metrics:** Accuracy, ROC-AUC, F1 Score, Precision, Recall
- **Classification threshold:** 0.5 throughout the academic comparison

**Eight classifiers compared:**

| Tier | Models |
|---|---|
| Classical | Logistic Regression, Gaussian Naive Bayes, K-Nearest Neighbours, Decision Tree |
| Ensemble | Random Forest, Gradient Boosting |
| Deep Learning | Scikit-learn MLP Classifier, Keras Sequential ANN |

---

## Repository contents

- `Customer_Churn_Analysis.ipynb` — main analysis notebook containing data preprocessing, exploratory analysis, feature engineering, model training, evaluation, and business impact quantification

## Requirements

- Python 3.9 or higher
- pandas
- numpy
- scikit-learn
- tensorflow / keras
- matplotlib
- seaborn

Install requirements via:

```bash
pip install pandas numpy scikit-learn tensorflow matplotlib seaborn
```

## Reproducibility

To reproduce the analysis:

1. Clone this repository
2. Install the required packages (see Requirements above)
3. Download the dataset from Kaggle : https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset?select=customer_churn_dataset-training-master.csv
4. Open `Customer_Churn_Analysis.ipynb` in Jupyter Notebook or JupyterLab
5. Run cells sequentially from top to bottom

Note: some randomness in model training may lead to minor variation in final metrics. All reported results in the thesis use a fixed random seed (see notebook for the specific value).

## Acknowledgements

I want to thank my supervisor, Dr. Rajesh Kulkarni, for the weekly guidance and consistent feedback throughout this project. I also thank Liverpool John Moores University and the MSc Data Science programme for providing the academic foundation for this work.

---

## Contact

For questions about this work, please open an issue on this repository or contact via the GitHub profile:

https://github.com/Divyansh-khari
