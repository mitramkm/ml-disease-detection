import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report


st.set_page_config(page_title="Heart Disease Classification", layout="wide")

def load_system_artifacts():
    # Model files
    model_paths = {
        "Logistic Regression": "model/logistic_regression.joblib",
        "Decision Tree": "model/decision_tree.joblib",
        "KNN": "model/knn.joblib",
        "Naive Bayes": "model/naive_bayes.joblib",
        "Support Vector Machine": "model/support_vector_machine.joblib"
    }
    
    loaded_models = {}
    for name, path in model_paths.items():
        loaded_models[name] = joblib.load(path)
        
    preprocessor = joblib.load('model/preprocessor.joblib')
    return loaded_models, preprocessor

def run_feature_engineering(dataframe):
    # Feature engineering - Same code used in model builing
    if 'id' in dataframe.columns:
        dataframe = dataframe.drop(columns=['id'])

    # 2. Replace string placeholders with numpy NaN values
    dataframe.replace(['', ' ', '?', 'NA', 'NaN'], np.nan, inplace=True)

    # 3. Drop 'ca' and 'slope' dynamically if present in the input schema
    for col in ['ca', 'slope']:
        if col in dataframe.columns:
            dataframe.drop(columns=[col], inplace=True)

    # 4. Cast target columns to explicit numeric types
    numeric_convert = ['trestbps', 'chol', 'thalch', 'oldpeak']
    for col in numeric_convert:
        if col in dataframe.columns:
            dataframe[col] = pd.to_numeric(dataframe[col], errors='coerce')

    # 5. Handle outliers for continuous columns using exact training quantile thresholds
    # (Prevents runtime crashes or mathematical anomalies on solitary test rows)
    if 'age' in dataframe.columns:
        dataframe['age'] = np.clip(dataframe['age'], 32.0, 76.0)
    if 'trestbps' in dataframe.columns:
        dataframe['trestbps'] = np.clip(dataframe['trestbps'], 92.38, 180.0)
    if 'chol' in dataframe.columns:
        dataframe['chol'] = np.clip(dataframe['chol'], 0.0, 458.0)
    if 'thalch' in dataframe.columns:
        dataframe['thalch'] = np.clip(dataframe['thalch'], 71.38, 182.0)
    if 'oldpeak' in dataframe.columns:
        dataframe['oldpeak'] = np.clip(dataframe['oldpeak'], -0.58, 4.0)

    # 6. Engineered Feature 1: Age Binning Categorization using explicit boundaries
    dataframe['age_group'] = pd.cut(
        dataframe['age'], 
        bins=[0, 35, 55, 70, 100], 
        labels=['Young', 'Middle-Aged', 'Senior', 'Elderly']
    )

    # 7. Engineered Feature 2: Metabolic Risk Index Score calculation
    # Uses the fixed historical dataset median value to handle divisions by zero safely
    historical_chol_median = 223.0 
    temp_chol = dataframe['chol'].replace(0, historical_chol_median)
    dataframe['bps_chol_ratio'] = dataframe['trestbps'] / temp_chol

    # 8. Engineered Feature 3: Cardiac Workload Index Estimate
    dataframe['max_hr_pct_reserve'] = dataframe['thalch'] / (220 - dataframe['age'])

    # 9. Pipeline Safety: Force string data types on categorical features 
    # to guarantee compatibility with SimpleImputer and OneHotEncoder
    categorical_convert = ['sex', 'dataset', 'cp', 'fbs', 'restecg', 'exang', 'thal', 'age_group']
    for col in categorical_convert:
        if col in dataframe.columns:
            dataframe[col] = dataframe[col].astype(str)

    return dataframe



def render_dashboard_metrics(y_true, y_pred, y_prob, selected_model_name):
    # Displays performance metrix 
    acc = accuracy_score(y_true, y_pred)
    auc_score = roc_auc_score(y_true, y_prob)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader(f"Summary Validation Scores: {selected_model_name}")
        perf_summary_df = pd.DataFrame({
            "Evaluation Metric Parameters": ["Accuracy Ratio", "Area Under Curve (AUC)", "Precision Core Score", "Recall Coverage Ratio", "F1 Balancing Metric", "Matthews Correlation (MCC)"],
            "Calculated Output Value": [f"{acc:.4f}", f"{auc_score:.4f}", f"{prec:.4f}", f"{rec:.4f}", f"{f1:.4f}", f"{mcc:.4f}"]
        })
        st.table(perf_summary_df)
        
    with col_right:
        st.subheader("🧩 Confusion Matrix Visualization Plot")
        cm = confusion_matrix(y_true, y_pred)
        fig_plot, ax_canvas = plt.subplots(figsize=(5, 3.8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', cbar=False,
                    xticklabels=['No Risk (0)', 'Heart Risk (1)'],
                    yticklabels=['No Risk (0)', 'Heart Risk (1)'], ax=ax_canvas)
        plt.ylabel('Actual Category Labels')
        plt.xlabel('Predicted Category Labels')
        st.pyplot(fig_plot)

# =====================================================================
# Main Method
# =====================================================================
st.title("Interactive Heart Disease Prediction")

st.sidebar.header("Upload Verification Data")
uploaded_file = st.sidebar.file_uploader("Upload the 'test_data.csv' file", type=["csv"])

if uploaded_file is not None:
    test_data = pd.read_csv(uploaded_file)
    st.success("Test dataset loaded successfully.")
    
    if 'target' not in test_data.columns:
        st.error("Corrupt file. Missing target format")
    else:
        try:
            # Load models
            models_dict, system_preprocessor = load_system_artifacts()
        except Exception as err:
            st.error(f"Failed to load model files: {err}")
            st.stop()
            
        # Run data transforms 
        test_data = run_feature_engineering(test_data)
        
        X_eval = test_data.drop(columns=['target'])
        y_eval = test_data['target']
        
        # Apply standard transformations
        X_eval_transformed = system_preprocessor.transform(X_eval)
        
        st.sidebar.header("Target Classifier Models")
        selected_model = st.sidebar.selectbox("Choose model for validation evaluation:", list(models_dict.keys()))
        
        active_clf = models_dict[selected_model]
        
        # Predict classes and output
        predictions = active_clf.predict(X_eval_transformed)
        probabilities = active_clf.predict_proba(X_eval_transformed)[:, 1] if hasattr(active_clf, "predict_proba") else predictions
        
        # Render metrics
        render_dashboard_metrics(y_eval, predictions, probabilities, selected_model)
        
        st.subheader("Display Classifiction report")
        st.text(classification_report(y_eval, predictions, zero_division=0))
else:
    st.info("Test file 'test_data.csv' files not found")
    
