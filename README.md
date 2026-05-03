# ❤️ Heart Disease Prediction Using Machine Learning

## 📌 Project Overview
This project develops a **machine learning-based system for heart disease prediction** using structured clinical data. It includes a full ML pipeline covering data preprocessing, feature selection, model training, evaluation, calibration, and explainability using SHAP.

The goal is to assist early diagnosis of heart disease using data-driven predictive modeling.

---

## 📄 Abstract (IEEE Style)
Heart disease is a leading cause of mortality worldwide, requiring early and accurate diagnosis for effective treatment. This study proposes a machine learning framework for heart disease prediction using clinical attributes. The pipeline includes data preprocessing, univariate feature filtering, recursive feature elimination with cross-validation (RFECV), and model training using decision tree-based methods. Model performance is evaluated using AUC-ROC, F1-score, confusion matrix, and Brier score for calibration assessment. Additionally, SHAP (SHapley Additive exPlanations) is used to interpret model predictions. Experimental results demonstrate strong predictive performance and generalization across internal and external validation datasets.

---

## 🎯 Objectives
- Develop an accurate heart disease prediction model  
- Perform feature selection using statistical + ML methods  
- Evaluate model performance using multiple metrics  
- Ensure interpretability using SHAP analysis  
- Validate generalization using external datasets  

---

## 📊 Dataset
- Source: UCI Cleveland Heart Disease Dataset  
- Type: Structured medical dataset  
- Features: Clinical attributes (age, cholesterol, chest pain type, etc.)  
- Target: Binary classification (Heart Disease: Yes / No)  

---

## ⚙️ Methodology

### 1. Data Preprocessing
- Handling missing values  
- Feature encoding  
- Feature scaling  

### 2. Feature Selection
- Stage 1: Univariate filtering (28 → 12 features)  
- Stage 2: RFECV (12 → 6 optimal features)  

### 3. Model Training
- Decision Tree Classifier  
- Cross-validation for robustness  

### 4. Evaluation Metrics
- Accuracy  
- ROC-AUC  
- F1-score  
- Confusion Matrix  
- Brier Score (Calibration quality)  

### 5. Explainability
- SHAP feature importance  
- Beeswarm plots  
- Feature contribution analysis  

---

## 📈 Results Summary

| Metric | Performance |
|--------|-------------|
| AUC-ROC | High (~0.95 internal) |
| F1-score | Strong performance |
| Calibration | Evaluated using Brier score |
| External Validation | Tested for generalization |

---

## 📊 Visualizations Included
- ROC Curves  
- Confusion Matrices  
- SHAP Bar & Beeswarm Plots  
- Reliability (Calibration) Diagrams  
- Feature Selection Curves  
- Dataset Shift Analysis (KS Test)  

---

## 🧠 Key Insights
- Chest pain type and exercise-induced angina are strong predictors  
- Feature selection improves model stability and performance  
- SHAP improves interpretability of predictions  
- Calibration analysis ensures probability reliability  

---

## 📁 Repository Structure
