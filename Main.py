from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

@app.get("/")
def home():
    return {"message": "API running!"}

@app.get("/hello")
def hello(name: str):
    return {"message": f"Hello {name}!"}

class InputData(BaseModel):
    age: int
    income: float

@app.post("/predict")
def predict(data: InputData):
    age = data.age
    income = data.income
    # Temporary fake model:
    prediction = age + income
    return {"prediction": prediction}