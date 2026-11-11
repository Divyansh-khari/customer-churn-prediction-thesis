# Customer Churn Prediction Through Comparative Machine Learning Analysis

MSc Data Science | Final Thesis | Liverpool John Moores University (LJMU)
Extended into a conference paper accepted at **DeSE 2026** (IEEE)

**Author:** Divyansh Khari
**Supervisor / co-author:** Dr. Rajesh Hanmanthrao Kulkarni
**Thesis submitted:** July 2026
**Paper:** *Comparing Classical, Ensemble and Neural Network Models for Telecom Customer Churn Prediction on a Large, Naturally Balanced Dataset*, DeSE 2026, 16–18 December 2026 (paper #1571375051)

---

## Overview

This repository contains the analysis code for my MSc dissertation on customer churn prediction and for the conference paper that grew out of it. The work compares eight machine learning classifiers from three families (classical, ensemble and neural) on a naturally balanced public telecom dataset of 440,832 customers, so no oversampling such as SMOTE is needed.

The study addresses four questions:

1. Which model family predicts churn most accurately when every model gets the same preprocessing, tuning budget and evaluation folds?
2. Do deeper neural networks help, and does a neural result hold when the network is rebuilt in a different library?
3. Which customer attributes are the strongest churn drivers across model families?
4. How does the decision threshold change the value of a churn model to an operator?

---

## Key findings (conference paper, revised protocol)

Results are the mean ± standard deviation over 5×2 cross-validation (ten train–test splits of about 200,000 customers each).

| Model | Accuracy (%) | F1 (%) | ROC-AUC |
|---|---|---|---|
| **Gradient Boosting** | **96.62 ± 0.05** | **96.97 ± 0.04** | **0.9921** |
| Keras network (128-64) | 96.35 ± 0.07 | 96.72 ± 0.06 | 0.9909 |
| Scikit-learn MLP (100-50) | 96.32 ± 0.05 | 96.69 ± 0.04 | 0.9907 |
| Random Forest | 96.19 ± 0.04 | 96.58 ± 0.03 | 0.9901 |
| Decision Tree | 95.08 ± 0.06 | 95.55 ± 0.06 | 0.9828 |
| K-Nearest Neighbours | 92.28 ± 0.05 | 92.94 ± 0.05 | 0.9606 |
| Logistic Regression | 89.48 ± 0.06 | 90.35 ± 0.06 | 0.9514 |
| Gaussian Naive Bayes | 78.56 ± 1.95 | 77.08 ± 2.66 | 0.9131 |

- **Best model:** tuned Gradient Boosting. Its lead over the networks is small (about 0.3 points) but consistent: it won on all ten folds (5×2cv paired t-test vs MLP: t = 6.28, p = 0.0015; vs Keras: t = 3.46, p = 0.018; Wilcoxon p = 0.002 in both cases).
- **Neural networks are competitive:** the Scikit-learn MLP and the Keras network are statistically indistinguishable from each other (Wilcoxon p = 0.19) and within 0.3 points of Gradient Boosting.
- **Depth:** across ten MLP configurations, the best two-layer network beat the best single-layer network by only 0.17 points, about one fold standard deviation. Extra depth gives at most marginal gains.
- **Feature drivers:** spend per tenure, support-call rate and monthly contract rank in the top three for Random Forest and Gradient Boosting and in the top four for the MLP.
- **Business value (illustrative):** with assumed costs of $100 per retained churner and $20 per wasted offer, the MLP's value on one split is about $4.73M at threshold 0.5 and rises as the threshold is lowered, because value is driven mainly by recall. These figures assume every flagged churner is retained and are not an operator's real numbers.

### How this differs from the thesis

The thesis (July 2026) reported the single-layer MLP as the best model (96.24% accuracy on one 80:20 split). In that analysis the MLP was tuned, but Gradient Boosting and Random Forest used fixed settings (for example, Gradient Boosting at depth 3). Following the DeSE reviewers' comments, every model was given the same tuning budget and evaluated with repeated cross-validation. With equal tuning, Gradient Boosting chose deeper trees and moved to the top. The neural network results themselves barely changed. Both notebooks are kept in this repository so the change can be traced.

---

## Methodology

The analysis follows the CRISP-DM framework.

**Dataset:** 440,832 customer records after removing one empty row (56.7% churn / 43.3% retained), 12 raw columns turned into 17 predictors.

**Features:** behavioural counts divided by tenure (spend, usage, support calls, payment delay), recency and engagement scores, a High_Risk flag, age bands, and one-hot encoded categories.

**Revised evaluation protocol (paper):**
- A stratified sample of 40,000 customers is held out and used only for hyperparameter tuning.
- Every model gets the same budget: 10 random configurations × 3-fold cross-validation.
- The remaining 400,832 customers are evaluated with 5×2 cross-validation, using the same folds for every model.
- The only steps that learn from data (the High_Risk medians and standardisation) are fitted on each training fold only, inside a Scikit-learn pipeline, so no information leaks from test to training data.
- Threshold 0.5. Metrics: accuracy, precision, recall, F1 and ROC-AUC (mean ± SD).
- The best model is compared with every other using the 5×2cv paired t-test (Dietterich, 1998) and the Wilcoxon signed-rank test.

**Original thesis protocol:** 80:20 stratified split (352,665 train / 88,167 test) with 5-fold cross-validation for model selection.

**Eight classifiers compared:**

| Family | Models |
|---|---|
| Classical | Logistic Regression, Gaussian Naive Bayes, K-Nearest Neighbours, Decision Tree |
| Ensemble | Random Forest, Gradient Boosting |
| Neural | Scikit-learn MLP Classifier, Keras Sequential ANN |

---

## Repository contents

- `Customer_Churn_Analysis.ipynb`: the original thesis notebook (preprocessing, exploratory analysis, feature engineering, single-split model comparison, feature importance, threshold and business-impact analysis).
- `Revision_Experiments.ipynb`: the revised experiments for the DeSE 2026 paper (leak-free pipeline, equal tuning budget, 5×2 cross-validation, paired significance tests, MLP architecture comparison and the Keras network on the same splits). It is self-contained and needs only the dataset CSV.

## Requirements

- Python 3.9 or higher
- pandas, numpy, scipy
- scikit-learn
- tensorflow / keras (only for the Keras network)
- matplotlib, seaborn

```bash
pip install pandas numpy scipy scikit-learn tensorflow matplotlib seaborn
```

## Reproducibility

1. Clone this repository.
2. Install the requirements above.
3. Download the dataset from Kaggle: https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset?select=customer_churn_dataset-training-master.csv and save it as `Training-data.csv` in the same folder as the notebooks.
4. To reproduce the paper's results, open `Revision_Experiments.ipynb` and run all cells from top to bottom. On a 6-core laptop with 16–32 GB RAM the full run takes about 2–3 hours, mostly for Gradient Boosting and KNN. Results are saved to `rev_*.csv` files as the notebook runs.
5. To reproduce the thesis results, run `Customer_Churn_Analysis.ipynb` from top to bottom.

All random seeds are fixed (seed 42 for tuning and models; seeds 0–4 for the five 5×2 repetitions). Small differences can still appear across library versions and hardware. The paper used Scikit-learn 1.7.2 and TensorFlow 2.21.0.

## Citation

If you use this work, please cite:

> D. Khari and R. H. Kulkarni, "Comparing Classical, Ensemble and Neural Network Models for Telecom Customer Churn Prediction on a Large, Naturally Balanced Dataset," in *Proc. DeSE 2026*, 2026.

## Acknowledgements

I want to thank my supervisor, Dr. Rajesh Kulkarni, for the weekly guidance and consistent feedback throughout this project, and the DeSE 2026 reviewers, whose comments led to the revised evaluation protocol. I also thank Liverpool John Moores University and the MSc Data Science programme for providing the academic foundation for this work.

---

## Contact

For questions about this work, please open an issue on this repository or contact me via my GitHub profile:

https://github.com/Divyansh-khari