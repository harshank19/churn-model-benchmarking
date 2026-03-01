import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
'''
df_ada = pd.read_csv(r"Classification_Reports\\AdaBoost_Report.csv",index_col=0)
df_log_reg = pd.read_csv(r"Classification_Reports\\LogReg_Report.csv",index_col=0)
df_dt = pd.read_csv(r"Classification_Reports\\DT_Report.csv",index_col=0)
df_grad_boost = pd.read_csv(r"Classification_Reports\\Grad_Boost_Report.csv",index_col=0)
df_rf = pd.read_csv(r"Classification_Reports\\RandomForest_Report.csv",index_col=0)
df_nb = pd.read_csv(r"Classification_Reports\\NaiveBayes_Report.csv",index_col=0)
df_grad_desc = pd.read_csv(r"Classification_Reports\\Grad_Desc_Report.csv",index_col=0)


def extract_metrics(report, model_name):
    df = pd.DataFrame([{"Model":model_name, "Accuracy": report.loc['accuracy','precision'],
                       "Minority_Precision":report.loc['1', "precision"], "Minority_Recall": report.loc['1', "recall"],
                       "Minority_F1":report.loc['1', "f1-score"], "Majority_Precision":report.loc['0', "precision"],
                       "Majority_Recall": report.loc['0', "recall"], "Majority_F1": report.loc['0', "f1-score"],
                       "Macro_Avg_Precision":report.loc['macro avg', "precision"], "Macro_Avg_Recall": report.loc['macro avg', "recall"],
                       "Macro_Avg_F1":report.loc['macro avg', "f1-score"], "Weighted_Avg_Precision":report.loc['weighted avg', "precision"],
                       "Weighted_Avg_Recall": report.loc['weighted avg', "recall"], "Weighted_Avg_F1":report.loc['weighted avg', "f1-score"]}])
    return df

model_names = ["Logistic Regression", "Bernoulli Naive Bayes", "Stochastic Gradient Descent", "Decision Tree", "Random Forest", "AdaBoost", "Gradient Boosting"]
report_list = [df_log_reg, df_nb, df_grad_desc, df_dt, df_rf, df_ada, df_grad_boost]
df_list = []

for i in range(len(report_list)):
    df_list.append(extract_metrics(report_list[i], model_names[i]))

final_df = pd.concat(df_list, ignore_index=True)
print(final_df.iloc[0])

#final_df.to_csv("Model_comparison.csv", index=False)
'''

df = pd.read_csv("Classification_Reports\\Model_comparison.csv")
#for i in range(len(df)):
#    print(df.loc[i])

print(df)

model_abbreviations = ["LR", "BNB", "SGD", "DT", "RF", "AB", "GB"]
plt.figure(figsize=(9, 5))
plt.scatter(df["Minority_Recall"], df["Minority_Precision"], marker="*", s=100)
for i in range(len(model_abbreviations)):
    plt.annotate(model_abbreviations[i], (df["Minority_Recall"][i], df["Minority_Precision"][i]+0.01))

plt.xlabel("Minority Recall")
plt.ylabel("Minority Precision")
plt.title("Minority Precision vs. Recall for Different Models")
#plt.ylim(0.8949, 0.8952)
#plt.xlim(0.91525, 0.9155)
#plt.savefig("Minority_Precision_vs_Recall.png", dpi=600)
plt.show()

df1 = df.copy()
df1["Model_abb"] = model_abbreviations
df2 = df1.sort_values("Accuracy", ascending=False)

plt.bar(df2['Model_abb'], df2['Accuracy'])
plt.xticks(rotation=45)
plt.ylabel("Accuracy Score")
plt.title("Accuracy by Model")
plt.show()

df3 = df1.sort_values(by="Minority_F1", ascending=False)

plt.figure()
plt.bar(df3["Model_abb"], df3["Minority_F1"])
plt.xticks(rotation=45)
plt.ylabel("Minority F1 Score")
plt.title("Minority Class F1 Score by Model")
plt.show()

df4 = df1.sort_values(by="Minority_Recall", ascending=False)

plt.figure()
plt.bar(df4["Model_abb"], df4["Minority_Recall"])
plt.xticks(rotation=45)
plt.ylabel("Minority Recall")
plt.title("Minority Class Recall by Model")
plt.show()


plt.figure()
plt.scatter(df["Accuracy"], df["Minority_F1"])

for i, txt in enumerate(df["Model"]):
    plt.annotate(txt,(df["Accuracy"][i], df["Minority_F1"][i]))

plt.xlabel("Accuracy")
plt.ylabel("Minority F1")
plt.title("Accuracy vs Minority F1")
plt.show()

plt.figure()
plt.bar(df1["Model_abb"], df1["Weighted_Avg_F1"], color="crimson", alpha=0.8, label="Weighted Avg")
plt.bar(df1["Model_abb"], df1["Macro_Avg_F1"], color="green", alpha=0.5, label="Macro Avg")
plt.xticks(rotation=45)
plt.title("Macro vs Weighted F1 Score Comparison")
plt.legend()
plt.show()

plt.figure()
plt.bar(df1["Model_abb"], df1["Weighted_Avg_F1"])
plt.xticks(rotation=45)
plt.title("Weighted F1 Score Comparison")
plt.show()

df1.set_index("Model_abb")[["Macro_Avg_F1","Weighted_Avg_F1"]].plot(kind="bar")
plt.xticks(rotation=45)
plt.show()

df1.set_index("Model_abb")[["Macro_Avg_Recall","Weighted_Avg_Recall"]].plot(kind="bar")
plt.xticks(rotation=45)
plt.show()

df1.set_index("Model_abb")[["Majority_Recall","Minority_Recall"]].plot(kind="bar")
plt.xticks(rotation=45)
plt.show()

x_axis = np.arange(0, len(df1["Model_abb"]), step=1)
plt.plot(x_axis, df1["Minority_Recall"], label="Minority Recall")
plt.plot(x_axis, df1["Majority_Recall"], label="Majority Recall")
plt.xlabel("Model")
plt.xticks(x_axis, labels=df1["Model_abb"], rotation=45)
plt.legend()
plt.show(