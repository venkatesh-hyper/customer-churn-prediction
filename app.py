import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import kagglehub
import joblib
import os

    

# Set page title and layout
st.set_page_config(page_title="Churn Prediction Dashboard", layout="wide")
st.title("Churn Prediction Dashboard")

st.write("""
This dashboard predicts customer churn based on various features using different machine learning models.
Explore the data, model performance, and make predictions for new customers.
""")

# --- Data Loading and Preprocessing ---
@st.cache_data
def load_data():
    try:
        path = kagglehub.dataset_download("shrutimechlearn/churn-modelling")
        file_path = os.path.join(path, 'Churn_Modelling.csv')
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

df = load_data()

if df is not None:
    st.sidebar.header("Data Exploration")
    if st.sidebar.checkbox("Show raw data"):
        st.subheader("Raw Data")
        st.dataframe(df.head())

    st.sidebar.header("Preprocessing Steps")
    st.sidebar.write("""
    - Label Encoding for 'Gender'
    - One-Hot Encoding for 'Geography'
    - Feature Scaling using StandardScaler
    """)

    label_encoder = LabelEncoder()
    df['Gender'] = label_encoder.fit_transform(df['Gender'])
    df = pd.get_dummies(df, columns=['Geography'], drop_first=True)

    features = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard",
                "IsActiveMember", "EstimatedSalary", "Gender", "Geography_Spain", "Geography_Germany"]
    X = df[features]
    y = df["Exited"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

def load_models():
    model_dir = 'models'
    models = {
        'Random Forest': joblib.load(os.path.join(model_dir, 'random_forest_model.joblib')),
        'Logistic Regression': joblib.load(os.path.join(model_dir, 'logistic_regression_model.joblib')),
        'SVM': joblib.load(os.path.join(model_dir, 'svm_model.joblib')),
        'KNN': joblib.load(os.path.join(model_dir, 'knn_model.joblib')),
        'Gradient Boosting': joblib.load(os.path.join(model_dir, 'gradient_boosting_model.joblib')),
         }
    return models

trained_models = load_models()

    # --- Model Evaluation ---
st.sidebar.header("Model Evaluation")
selected_model_name = st.sidebar.selectbox("Select Model for Evaluation", list(trained_models.keys()))
    
   

# Two columns layout to keep plots compact horizontally
col1, col2 = st.columns(2)

with col1:
    # Pie chart for Exited (Churn)
    fig, ax = plt.subplots(figsize=(4, 4))
    df['Exited'].value_counts().plot.pie(
        autopct='%1.1f%%',
        startangle=90,
        colors=['#4CAF50', '#F44336'],
        labels=['Stayed', 'Churned'],
        ax=ax
    )
    ax.set_ylabel('')
    plt.setp(ax.texts, size=10)
    st.pyplot(fig)

with col2:
    # Histogram for Tenure
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(df['Tenure'], bins=30, color='#1976D2', alpha=0.7)
    ax.set_title('Customer Tenure Distribution', fontsize=12)
    ax.set_xlabel('Tenure (months)', fontsize=10)
    ax.set_ylabel('Number of Customers', fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=8)
    st.pyplot(fig)

with col1:
    # Boxplot for Balance by Exited
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.boxplot(x='Exited', y='Balance', data=df, ax=ax, palette=['#4CAF50', '#F44336'])
    ax.set_title('Balance by Churn Status', fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=8)
    st.pyplot(fig)

with col2:
    # Bar plot for NumOfProducts
    fig, ax = plt.subplots(figsize=(4, 3))
    df['NumOfProducts'].value_counts().plot.bar(color='#2196F3', ax=ax)
    ax.set_title('Number of Products Distribution', fontsize=12)
    ax.set_xlabel('Number of Products', fontsize=10)
    ax.set_ylabel('Number of Customers', fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=8)
    st.pyplot(fig)

with col1:
    # Boxplot for EstimatedSalary by Exited
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.boxplot(x='Exited', y='EstimatedSalary', data=df, ax=ax, palette=['#4CAF50', '#F44336'])
    ax.set_title('Estimated Salary by Churn Status', fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=8)
    st.pyplot(fig)

with col2:
    # Bar plot for HasCrCard
    fig, ax = plt.subplots(figsize=(4, 3))
    df['HasCrCard'].value_counts().plot.bar(color='#FF9800', ax=ax)
    ax.set_title('Has Credit Card Distribution', fontsize=12)
    ax.set_xlabel('Has Credit Card', fontsize=10)
    ax.set_ylabel('Number of Customers', fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=8)
    st.pyplot(fig)

with col1:
    # Boxplot for Age by Exited
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.boxplot(x='Exited', y='Age', data=df, ax=ax, palette=['#4CAF50', '#F44336'])
    ax.set_title('Age by Churn Status', fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=8)
    st.pyplot(fig)


    if selected_model_name:
        st.subheader(f"Evaluation Results for {selected_model_name}")
        current_model = trained_models[selected_model_name]
        y_pred = current_model.predict(X_test_scaled)

        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"**Accuracy:** {accuracy:.4f}")

        st.write("**Confusion Matrix**")
        conf_matrix = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'Confusion Matrix for {selected_model_name}')
        st.pyplot(fig)

        st.write("**Classification Report**")
        class_report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(class_report).transpose()
        st.dataframe(report_df)

    # --- Feature Importance ---
    st.subheader("Feature Importance")
    current_model = trained_models[selected_model_name]
    if selected_model_name in ['Random Forest', 'Gradient Boosting'] and hasattr(current_model, 'feature_importances_'):
        importance = current_model.feature_importances_
        indices = np.argsort(importance)[::-1]
        names = [features[i] for i in indices]

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=importance[indices], y=names, ax=ax, palette='viridis')
        ax.set_title(f"Feature Importance for {selected_model_name}")
        st.pyplot(fig)

    elif selected_model_name == 'Logistic Regression' and hasattr(current_model, 'coef_'):
        coefficients = current_model.coef_[0]
        coef_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': coefficients})
        coef_df = coef_df.sort_values(by='Coefficient', ascending=False)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Coefficient', y='Feature', data=coef_df, ax=ax, palette='coolwarm')
        ax.set_title("Logistic Regression Feature Coefficients")
        st.pyplot(fig)

    else:
        st.info("Feature importance not available for the selected model.")
        
    # --- Model Comparison ---
    st.subheader("Model Comparison")
    def get_metrics_from_report(report_dict):
        return report_dict.get('macro avg', {'precision': np.nan, 'recall': np.nan, 'f1-score': np.nan})

    comparison_data = []
    for name, model in trained_models.items():
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        class_report_dict = classification_report(y_test, y_pred, output_dict=True)
        metrics = get_metrics_from_report(class_report_dict)

        comparison_data.append({
            'Model': name,
            'Accuracy': accuracy,
            'Precision': metrics.get('precision'),
            'Recall': metrics.get('recall'),
            'F1-score': metrics.get('f1-score')
        })

    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df.set_index('Model').style.highlight_max(axis=0))

    # --- Make Prediction ---
    st.sidebar.header("Make a Prediction")
    with st.sidebar.form("prediction_form"):
        st.write("Enter customer details:")
        creditscore = st.number_input("Credit Score", 300, 850, 650)
        age = st.number_input("Age", 18, 92, 40)
        tenure = st.number_input("Tenure", 0, 10, 5)
        balance = st.number_input("Balance", 0.0, 250000.0, 50000.0)
        numofproducts = st.number_input("Number of Products", 1, 4, 1)
        hascrcard = st.selectbox("Has Credit Card", [0, 1], format_func=lambda x: "Yes" if x else "No")
        isactivemember = st.selectbox("Is Active Member", [0, 1], format_func=lambda x: "Yes" if x else "No")
        estimatedsalary = st.number_input("Estimated Salary", 0.0, 200000.0, 75000.0)
        gender = st.selectbox("Gender", ["Male", "Female"])
        geography = st.selectbox("Geography", ["France", "Spain", "Germany"])

        predict_button = st.form_submit_button("Predict Churn")

    if predict_button:
        gender_val = 1 if gender == "Male" else 0
        geo_spain = 1 if geography == "Spain" else 0
        geo_germany = 1 if geography == "Germany" else 0

        input_data = np.array([[creditscore, age, tenure, balance, numofproducts,
                                hascrcard, isactivemember, estimatedsalary,
                                gender_val, geo_spain, geo_germany]])

        input_scaled = scaler.transform(input_data)
        prediction = current_model.predict(input_scaled)[0]
        prediction_proba = current_model.predict_proba(input_scaled)[0][1]

        st.sidebar.subheader("Prediction Result")
        if prediction == 1:
            st.sidebar.error(f"Customer is likely to churn. (Probability: {prediction_proba:.2f})")
        else:
            st.sidebar.success(f"Customer is likely to stay. (Probability: {1 - prediction_proba:.2f})")
