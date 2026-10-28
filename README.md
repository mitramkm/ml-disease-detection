# ml-disease-detection
ML models for Heart Disease classification


### a. Problem Statement
The objective of this assignment is to implement an end-to-end Machine Learning classification and deployment workflow. The task involves cleaning a clinical cardiovascular registry dataset, engineering diagnostic risk proxies, training and evaluating five distinct classification architectures (`Logistic Regression`, `Decision Tree`, `KNN`, `Naive Bayes`, and `Support Vector Machine`), serializing the model pipelines, and deploying them to a public production environment via Streamlit Community Cloud.

---

### b. Dataset Description
- **Dataset Source:** Provided clinical multi-center cardiovascular records registry.
- **Problem Formulation:** Binary classification problem mapped from the multi-class `num` label column (where `target = 0` denotes a healthy patient profile and `target = 1` denotes the presence of heart disease risk).
- **Instance Size:** 920 total customer/patient rows.
- **Feature Vector Dimension Size:** 14 predictive biometric attributes columns + 1 target attribute.
- **Predictive Columns Used:** age, sex, dataset location source, cp (chest pain classification type), trestbps (resting blood pressure markers), chol (serum cholesterol indicators), fbs (fasting blood sugar level limits), restecg (resting electrocardiographic readings), thalch (maximum heart rate targets achieved), exang (exercise-induced angina indicators), oldpeak (ST depression parameters), and thal (thallium stress test categories).
- **Engineered Features Added:** `age_group` (discretized non-linear lifetime buckets), `bps_chol_ratio` (metabolic stress load proxy interaction indicator), and `max_hr_pct_reserve` (cardiac volume reserve ratio metric).

---

### c. GitHub Repository Link
- **GitHub Repository URL:** https://github.com/mitramkm/ml-disease-detection/

---

### d. Models Used & Performance Evaluation Matrix

#### 1. Evaluation Metrics Comparison Table
Below is the evaluation metrics matrix computed over the isolated, stratified 20% validation split (`test_data.csv`) that the classifiers have never seen before:

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 0.858696 | 0.927068 | 0.851852 | 0.901961 | 0.876190 | 0.713524 |
| Decision Tree | 0.798913 | 0.840328 | 0.849462 | 0.774510 | 0.810256 | 0.600236 |
| KNN | 0.831522 | 0.903276 | 0.831776 | 0.872549 | 0.851675 | 0.657972 |
| Naive Bayes | 0.836957 | 0.913259 | 0.839623 | 0.872549 | 0.855769 | 0.669083 |
| Support Vector Machine | 0.836957 | 0.912064 | 0.815789 | 0.911765 | 0.861111 | 0.671257 |

#### 2. Model Performance Observations Analysis

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Best overall performer. It works perfectly here because the clean numbers and categories create a clear, straight-line boundary to split healthy and high-risk patients accurately. |
| **Decision Tree** | Simplest model. It creates simple "yes/no" rules that are easy for to understand. However, it is a bit too rigid, making it less accurate than the other models. |
| **KNN** | Good at finding similar patients. It checks groups of similar patients to make a guess. It is stable but gets slightly confused by missing data values in the dataset. |
| **Naive Bayes** | Fast and accurate. Even though it assumes all symptoms are completely independent of each other, it beats the Decision Tree and showed a good scores. |
| **Support Vector Machine** | Best at capturing high-risk patients. It handles complex symptoms very smoothly. It scored the highest in catching actual heart disease cases (highest Recall). |

#### 3. Overall Dataset Winner Verdict
**Logistic Regression** is selected as the clear winning model framework for this heart disease dataset configuration. It achieves the highest overall performance indicators across all primary tracking parameters, maximizing **Accuracy (0.858696)**, **AUC Score (0.927068)**, and **MCC Score (0.713524)**.

