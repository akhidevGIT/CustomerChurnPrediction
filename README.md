# Customer Churn Prediction – End-to-End Machine Learning System
A fully containerized, production-grade Churn Prediction system built with FastAPI, Streamlit, XGBoost, scikit-learn, SHAP, and automated with GitHub Actions CI/CD.
## Key Features
- **Automated ELT Pipeline**
Load → clean → transform → encode → split data

- **Training Pipeline**
Trains multiple models (Logistic Regression, Random Forest, XGBoost)

- **Model Selection**
Automatically selects the best model based on evaluation metrics

- **SHAP Explainability**
Per-customer insights + global feature importance

- **FastAPI Prediction Service**
Endpoints for single prediction, batch prediction, and SHAP values

- **User-Friendly Streamlit UI**
Upload CSV → get predictions → visualize SHAP explanations

- **CI/CD with GitHub Actions**
On push: run training → package artifacts → build images → push to Docker Hub

## Local Setup (without Docker)
### Clone the repository
```bash
git clone https://github.com/akhidevGIT/CustomerChurnPrediction.git
cd project
```
### Install dependencies
#### Backend
```bash
pip install -r requirements_api.txt
```
#### Frontend
```bash
pip install -r requirements_ui.txt
```
### Train model
```bash
python pipelines/train_pipeline.py
```
### Run backend
```bash
uvicorn src.api.main:app --reload
```
### Run frontend
```bash
streamlit run ui/app.py
```

## 📚 Learning Highlights
This project demonstrates:

- ML engineering best practices

- Automated training pipelines

- CI/CD automation

- API design (FastAPI)

- Interactive dashboards (Streamlit)

- Deployment using Docker

## 🧑‍💻 Author

Akhila Devarapalli
Data Scientist | GenAI Enthusiast
