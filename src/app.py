import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Inference service for predicting customer churn in a cloud platform engineering environment",
    version="1.0.0"
)

# Load trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "churn_model.pkl")

class CustomerData(BaseModel):
    tenure: int = Field(..., ge=0, le=100, description="Customer tenure in months (e.g. 12)")
    monthly_charges: float = Field(..., ge=0.0, description="Monthly charges in USD (e.g. 65.50)")
    total_charges: float = Field(..., ge=0.0, description="Total accumulated charges (e.g. 786.00)")
    contract_type: int = Field(..., ge=0, le=2, description="0: Month-to-month, 1: One year, 2: Two year")
    tech_support: int = Field(..., ge=0, le=1, description="0: No, 1: Yes")
    online_security: int = Field(..., ge=0, le=1, description="0: No, 1: Yes")

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "Customer Churn Prediction Service",
        "version": "1.0.0"
    }

@app.get("/health")
def health_check():
    model_loaded = os.path.exists(MODEL_PATH)
    return {
        "status": "healthy" if model_loaded else "model_missing",
        "model_file_exists": model_loaded
    }

@app.post("/predict")
def predict_churn(customer: CustomerData):
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(
            status_code=503,
            detail="Model is not yet trained. Run src/train.py first."
        )
    
    model = joblib.load(MODEL_PATH)
    
    input_df = pd.DataFrame([{
        "tenure": customer.tenure,
        "monthly_charges": customer.monthly_charges,
        "total_charges": customer.total_charges,
        "contract_type": customer.contract_type,
        "tech_support": customer.tech_support,
        "online_security": customer.online_security
    }])
    
    prediction = int(model.predict(input_df)[0])
    churn_probability = float(model.predict_proba(input_df)[0][1])
    
    return {
        "churn_prediction": prediction,
        "churn_status": "Will Churn" if prediction == 1 else "Will Retain",
        "churn_probability": round(churn_probability, 4)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.app:app", host="0.0.0.0", port=8000, reload=True)
