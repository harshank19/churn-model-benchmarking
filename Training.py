import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
#from sklearn.neighbors import KNeighborsClassifier      #DOESN'T WORK!!!
#from sklearn.svm import SVC                             #DOESN'T WORK!!!
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, classification_report,\
roc_curve, roc_auc_score, precision_recall_curve, average_precision_score


df = pd.read_parquet("0000.parquet")
df = df.drop(["customer_id", "snapshot_date"],axis=1)


df["gender"] = df["gender"].map({"male":0, "female":1})
binary_cols = ["has_savings_account", "has_current_account", "has_loan", "has_debit_card", "has_credit_card",
               "has_wallet", "has_investment", "dormant_flag"]

for i in binary_cols:
    df[i] = df[i].map({True:1, False:0})

encoder_kyc = OneHotEncoder(sparse_output=False, drop=None)
encoder_state = OneHotEncoder(sparse_output=False, drop=None)

encoded_array_kyc = encoder_kyc.fit_transform(df[["kyc_tier"]])
encoded_array_state = encoder_state.fit_transform(df[["state"]])

encoded_df_kyc = pd.DataFrame(
    encoded_array_kyc,
    columns=encoder_kyc.get_feature_names_out(["kyc_tier"]),
    index=df.index
)

encoded_df_state = pd.DataFrame(encoded_array_state,columns=encoder_state.get_feature_names_out(["state"]), index=df.index)

df = df.drop(["kyc_tier", "state"], axis=1)
df = pd.concat([df, encoded_df_kyc, encoded_df_state], axis=1)

y = df["churn_30d"].map({True:1, False:0})
X = df.drop(["churn_30d","churn_90d"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

## Logistic Regression:

model_log_reg = LogisticRegression(max_iter = 1000, random_state=14758)
model_log_reg.fit(X_train, y_train)
y_pred_log_reg = model_log_reg.predict(X_test)
score_log_reg = accuracy_score(y_test, y_pred_log_reg)
print("Logistic Regression Score:", score_log_reg)
#report_dict = classification_report(y_test, y_pred_log_reg, output_dict=True)
#df = pd.DataFrame(report_dict).transpose()
#df.to_csv('LogReg_Report.csv')


## Support Vector Machines:
'''
model_svm = SVC(random_state=14758)
model_svm.fit(X_train, y_train)
y_pred_svm = model_svm.predict(X_test)
score_svm = accuracy_score(y_test, y_pred_svm)
print(score_svm)
'''

## K Nearest Neighbours:
'''
model_knn = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
model_knn.fit(X_train, y_train)
y_pred_knn = model_knn.predict(X_test)
score_knn = accuracy_score(y_test, y_pred_knn)
print("K Nearest Neighbors:", score_knn)
report_dict = classification_report(y_test, y_pred_knn, output_dict=True)
df = pd.DataFrame(report_dict).transpose()
df.to_csv('KNN_Report.csv')
'''

## Naive Bayes - BernoulliNB:

model_nb = BernoulliNB(alpha=1.0)
model_nb.fit(X_train, y_train)
y_pred_nb = model_nb.predict(X_test)
score_nb = accuracy_score(y_test, y_pred_nb)
print("Bernoulli NB Score:", score_nb)
#report_dict = classification_report(y_test, y_pred_nb, output_dict=True)
#df = pd.DataFrame(report_dict).transpose()
#df.to_csv('NaiveBayes_Report.csv')

## Decision Tree:

model_dt = DecisionTreeClassifier(random_state=14758, max_depth=5)
model_dt.fit(X_train, y_train)
y_pred_dt = model_dt.predict(X_test)
score_dt = accuracy_score(y_test, y_pred_dt)
print("Decision Tree Score:", score_dt)
#report_dict = classification_report(y_test, y_pred_dt, output_dict=True)
#df = pd.DataFrame(report_dict).transpose()
#df.to_csv('DT_Report.csv')

## Stochastic Gradient Descent:

model_grad_desc = SGDClassifier(random_state=14758) #loss = "hinge"(SVM), "log_loss"(logistic)---> default is "hinge"
model_grad_desc.fit(X_train, y_train)
y_pred_grad_desc = model_grad_desc.predict(X_test)
score_grad_desc = accuracy_score(y_test, y_pred_grad_desc)
print("Stochastic Grad Desc Score:", score_grad_desc)
#report_dict = classification_report(y_test, y_pred_grad_desc, output_dict=True)
#df = pd.DataFrame(report_dict).transpose()
#df.to_csv('Grad_Desc_Report.csv')

## Random Forest:

model_rf = RandomForestClassifier(n_estimators=100, random_state=14758)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)
score_rf = accuracy_score(y_test, y_pred_rf)
print("Random Forest Score:", score_rf)
#report_dict = classification_report(y_test, y_pred_rf, output_dict=True)
#df = pd.DataFrame(report_dict).transpose()
#df.to_csv('RandomForest_Report.csv')

## Adaptive Boosting:

model_ada_boost = AdaBoostClassifier(n_estimators=50, random_state=14758)
model_ada_boost.fit(X_train, y_train)
y_pred_ada_boost = model_ada_boost.predict(X_test)
score_ada_boost = accuracy_score(y_test, y_pred_ada_boost)
print("AdaBoost Score:", score_ada_boost)
#report_dict = classification_report(y_test, y_pred_ada_boost, output_dict=True)
#df = pd.DataFrame(report_dict).transpose()
#df.to_csv('AdaBoost_Report.csv')

## Gradient Boosting:

model_grad_boost = GradientBoostingClassifier(n_estimators=100, random_state=14758)
model_grad_boost.fit(X_train, y_train)
y_pred_grad_boost = model_grad_boost.predict(X_test)
score_grad_boost = accuracy_score(y_test, y_pred_grad_boost)
print("Grad Boost Score:", score_grad_boost)
#report_dict = classification_report(y_test, y_pred_grad_boost, output_dict=True)
#df = pd.DataFrame(report_dict).transpose()
#df.to_csv('Grad_Boost_Report.csv')

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
plt.savefig("PR_Curve.png", dpi=600)
plt.show()