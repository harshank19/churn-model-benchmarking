from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, classification_report,\
roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

models = {
    "Logistic Regression": model_log_reg,
    "Bernoulli NB": model_nb,
    "SGD": model_grad_desc,
    "Decision Tree": model_dt,
    "Random Forest": model_rf,
    "AdaBoost": model_ada_boost,
    "Gradient Boosting": model_grad_boost
}


plt.figure(figsize=(8, 6))

for name, model in models.items():

    # Get probability or decision scores
    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X_test)[:, 1]
    else:
        y_scores = model.decision_function(X_test)

    fpr, tpr, _ = roc_curve(y_test, y_scores)
    auc = roc_auc_score(y_test, y_scores)

    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")

# Random baseline
plt.plot([0, 1], [0, 1], linestyle='--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("ROC_Curve.png", dpi=600)
plt.show()

plt.figure(figsize=(8, 6))

for name, model in models.items():

    # Get probability or decision scores
    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X_test)[:, 1]
    else:
        y_scores = model.decision_function(X_test)

    precision, recall, _ = precision_recall_curve(y_test, y_scores)
    ap = average_precision_score(y_test, y_scores)

    plt.plot(recall, precision, label=f"{name} (AP = {ap:.3f})")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve Comparison")
plt.legend()
plt.tight_layout()
plt.show()