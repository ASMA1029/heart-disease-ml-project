# ❤️ Heart Disease Prediction Using Machine Learning

## 📌 Overview
This project builds a machine learning system for predicting heart disease using clinical data. It includes full ML pipeline development with preprocessing, feature selection, model training, evaluation, calibration, and explainability (SHAP).

---

## 🎯 Models Used
- Logistic Regression  
- Decision Tree  
- Random Forest  
- XGBoost  

---

## ⚙️ Pipeline
- Data preprocessing  
- Feature selection (Univariate + RFECV)  
- Model training with cross-validation  
- Evaluation using multiple metrics  
- Model explainability using SHAP  

---

## 📊 Evaluation Metrics
- Accuracy  
- ROC-AUC  
- F1 Score  
- Confusion Matrix  
- Brier Score (Calibration)  

---

## 📈 Key Features
- SHAP explainability plots  
- ROC curve comparison  
- Calibration analysis  
- External validation testing  
- Covariate shift analysis  

---

## 🧠 Key Insights
- Chest pain type is a strong predictor  
- Ensemble models outperform single models  
- Calibration improves reliability  
- SHAP improves interpretability  

---

## 📁 Project Files
- Python training pipeline  
- Trained ML models (.joblib)  
- Evaluation reports  
- JSON result logs  
- Visualization plots  

---

## 🚀 How to Run
```bash
pip install -r requirements.txt
python heart_disease_project_v2.py
