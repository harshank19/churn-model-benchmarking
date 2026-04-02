from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import create_model

app = FastAPI()

model_path = "Trained_models/model_rf.pkl"
model = joblib.load(model_path)

features = model.feature_names_in_

# Dynamically create InputData schema
InputData = create_model(
    "InputData",
    **{feature: (float, ...) for feature in features}
)

@app.get("/")
def home():
    return {"message": "Welcome to the Random Forest Churn API!"}
@app.post("/predict")
def predict(data: InputData):
    df = pd.DataFrame([data.model_dump()])
    prediction = model.predict(df)
    return {"prediction": int(prediction[0])}

