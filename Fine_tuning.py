import pandas as pd
import mlflow.sklearn
import scipy.stats as stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

filepath = r"C:\Users\hmnim\Desktop\OPINE Data Science\Python\ML1\Kaggle\BankingChurn\\0000.parquet"
df = pd.read_parquet(filepath)

targets = df[["churn_30d", "churn_90d"]]
df = df.drop(["customer_id", "snapshot_date", "churn_30d", "churn_90d"], axis=1)

num_cols = df.select_dtypes(include=['number']).columns.tolist()
bool_cols = df.select_dtypes(include=['bool']).columns.tolist()
cat_cols = df.select_dtypes(include=['category','object']).columns.tolist()

y = targets["churn_30d"]
X = df

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

preprocessor = ColumnTransformer(transformers=[('num', StandardScaler(), num_cols), ('cat', OneHotEncoder(), cat_cols)], remainder='passthrough')

pipeline = Pipeline([('preprocessor', preprocessor),('model', RandomForestClassifier())])

param_distributions = {
    "model__n_estimators": stats.randint(100, 600),
    "model__max_depth": [None] + list(range(5, 31, 5)),
    "model__min_samples_split": stats.randint(2, 15),
    "model__min_samples_leaf": stats.randint(1, 10),
    "model__max_features": ["sqrt", "log2"],
    "model__class_weight": [None, "balanced"]
}

random_search = RandomizedSearchCV(estimator=pipeline, param_distributions=param_distributions, cv=5, scoring="f1", n_jobs=1)

mlflow_switch = True #False
if mlflow_switch:
    # Set tracking server
    #mlflow.set_tracking_uri("http://127.0.0.1:5000")

    # Set experiment
    mlflow.set_experiment("Bank_Churn")

    with mlflow.start_run():
        random_search.fit(X_train, y_train)
        mlflow.log_params(random_search.best_params_)
        mlflow.log_metric("best_CV_score", random_search.best_score_)
        mlflow.sklearn.log_model(random_search.best_estimator_, artifact_path="best_RF_model")