# Churn Model Benchmarking

Imbalance-aware comparative evaluation of supervised machine learning models for customer churn prediction.

## Objective

To systematically compare multiple classification algorithms under severe class imbalance (~5% churn rate) and evaluate their minority-class detection performance.

## Models Evaluated

* Logistic Regression
* Naive Bayes
* SGD Classifier
* Decision Tree
* Random Forest
* Gradient Boosting
* AdaBoost

## Evaluation Strategy

* Precision, Recall, F1 (macro & weighted)
* Minority-class metrics
* ROC Curve
* Precision-Recall Curve
* Model ranking analysis

## Key Insight

Ensemble methods significantly outperform linear models in minority-class ranking and recall performance under heavy imbalance.

## Repository Structure

* `Training.py` – Model training pipeline
* `Evaluation.py` – Metric computation & performance comparison
* `Classification_Report_Evaluation.py` – Report generation and comparative summaries
