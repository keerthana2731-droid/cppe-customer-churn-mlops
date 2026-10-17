# Automated MLOps Pipeline for Customer Churn Prediction Using Jenkins and MLflow

This project implements an end-to-end MLOps and CI/CD platform for Customer Churn Prediction, fulfilling all academic requirements for the Cloud Product and Platform Engineering (CPPE) curriculum at SRM IST.

---

## Architecture Flow
```
Jira (Agile Tracking)
   │
   ▼
GitHub (Source Code Management)
   │
   ▼
Jenkins CI/CD Pipeline
   ├── Stage 1: Checkout Code
   ├── Stage 2: Install Dependencies (pip)
   ├── Stage 3: Automated Testing (pytest)
   ├── Stage 4: Model Training & Experiment Logging (MLflow)
   └── Stage 5: Deployment Validation & Health Check
   │
   ▼
Production REST API (FastAPI) & MLflow Tracking Server
```

---

## Quick Start Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Dataset & Train Model with MLflow
```bash
python src/train.py
```
This will:
- Generate `data/churn_data.csv` (1,000 synthetic customer records)
- Train a Random Forest classifier
- Log parameters and performance metrics into MLflow
- Save model weights to `models/churn_model.pkl`

### 3. Launch MLflow Tracking UI
```bash
mlflow ui
```
- Open [http://localhost:5000](http://localhost:5000)
- View parameters, accuracy/F1/ROC-AUC metrics, and model artifacts.
- **Take screenshots for Chapter 5.4, 6.4, and 8.5 of your report!**

### 4. Run Automated Unit & Integration Tests
```bash
pytest tests/ -v
```
- **Take screenshots of test pass results for Chapter 7.2 and 7.8.**

### 5. Launch FastAPI Inference Service
```bash
python -m uvicorn src.app:app --reload --port 8000
```
- Open Swagger Docs: [http://localhost:8000/docs](http://localhost:8000/docs)
- Test the `/predict` endpoint with sample customer parameters.
- **Take screenshots of Swagger UI for Chapter 6.5 and 6.6.**

### 6. Push to GitHub
```bash
git init
git add .
git commit -m "feat: initial commit for customer churn mlops pipeline"
git branch -M main
git remote add origin https://github.com/<your-username>/cppe-customer-churn-mlops.git
git push -u origin main
```
