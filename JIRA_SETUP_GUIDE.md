# Jira Setup Guide for CPPE Report

Use this guide to set up your free Jira Cloud board and take the screenshots required for **Chapter 5.1 (Jira Setup)** and **Chapter 7.7 (Defect Tracking using Jira)**.

---

## 1. Project Creation
- **Project Type:** Scrum (or Kanban)
- **Project Name:** Customer Churn MLOps
- **Project Key:** CHURN

---

## 2. User Stories for Sprint 1 (Chapter 5.1)
Create these issues in your Jira Backlog and start a 1-week or 2-week Sprint named **"Sprint 1 - Core MLOps Pipeline"**:

1. **CHURN-1 (Story):** *Dataset Ingestion and Preprocessing Pipeline*
   - **Description:** Implement automated ingestion for customer demographic and billing data with train-test split.
   - **Story Points:** 3
   - **Status:** Done

2. **CHURN-2 (Story):** *ML Model Training with MLflow Tracking*
   - **Description:** Train RandomForest classifier and track hyper-parameters (`n_estimators`, `max_depth`) and metrics (`accuracy`, `f1_score`, `roc_auc`) in MLflow.
   - **Story Points:** 5
   - **Status:** Done

3. **CHURN-3 (Story):** *FastAPI Inference Web Service*
   - **Description:** Build REST API endpoints `/health` and `/predict` for real-time customer churn prediction.
   - **Story Points:** 5
   - **Status:** Done

4. **CHURN-4 (Story):** *Continuous Integration & Jenkinsfile Pipeline*
   - **Description:** Configure automated CI stages: code checkout, dependency installation, pytest suite, model training, and smoke test.
   - **Story Points:** 8
   - **Status:** Done

---

## 3. Defect / Bug for Defect Tracking (Chapter 7.7)
Create a **Bug** ticket to show in your report:

- **Issue Type:** Bug
- **Key:** CHURN-5
- **Summary:** *Prediction API returns 500 error when model artifact file is missing*
- **Severity:** High / Critical
- **Steps to Reproduce:**
  1. Delete `models/churn_model.pkl`.
  2. Send POST request to `/predict`.
  3. API crashed instead of returning meaningful error message.
- **Resolution:** Updated `src/app.py` to check for model existence and return clean `HTTP 503` with warning message.
- **Status:** Closed / Resolved

---

## 4. Screenshots to Capture for Your Report
- [ ] **Screenshot 1:** Jira Scrum Board showing columns: *To Do*, *In Progress*, *Done* with the stories moved to *Done*. (For Chapter 5.1 & 6.6)
- [ ] **Screenshot 2:** Sprint Burndown Chart or Sprint completion report. (For Chapter 8.4)
- [ ] **Screenshot 3:** Defect / Bug Ticket (CHURN-5) details page showing status "Resolved". (For Chapter 7.7)
