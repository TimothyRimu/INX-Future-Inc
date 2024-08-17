import streamlit as st
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import RandomOverSampler

# Load and preprocess the dataset
@st.cache_data
def load_data():
    df = pd.read_excel('INX_Future_Inc_Employee_Performance_CDS_Project2_Data_V1.8.xlsx')
    df = df.drop('EmpNumber', axis=1)  # Dropping unnecessary columns
    
    # Encoding categorical variables using OrdinalEncoder
    categorical_cols = df.select_dtypes(include=['object']).columns
    encoder = OrdinalEncoder()
    df[categorical_cols] = encoder.fit_transform(df[categorical_cols])
    
    # Define features and target variable
    X = df.drop('PerformanceRating', axis=1)
    y = df['PerformanceRating']
    
    # Adjusting the target variable to start from 0
    y = y - y.min()
    
    # Balance the dataset using RandomOverSampler
    ros = RandomOverSampler(random_state=42)
    X, y = ros.fit_resample(X, y)
    
    return X, y

# Train the XGBoost model with GridSearchCV
@st.cache_resource
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize the XGBoost model and set up the parameter grid for tuning
    xgb_model = XGBClassifier(random_state=42)
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3]
    }
    
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    return best_model, accuracy, report

# Streamlit app
st.title("INX Future Inc. Employee Performance Evaluation")

st.write("""
This app uses a tuned XGBoost model with GridSearchCV to predict employee performance based on historical data.
You can use it to assess the potential performance of new employees.
""")

X, y = load_data()
model, accuracy, report = train_model(X, y)

st.write(f"Model Accuracy: {accuracy}")
st.text("Classification Report:")
st.text(report)

st.subheader("Predict Employee Performance")

# Collect user input for new employee data
input_data = {}
for col in X.columns:
    input_data[col] = st.number_input(f"{col}", value=0, step=1)

# Predict performance
input_df = pd.DataFrame([input_data])
prediction = model.predict(input_df)

st.write(f"Predicted Performance Rating: {prediction[0]}")


        
st.write("Developed for INX Future Inc.")
st.write("Copyright © 2024. All rights reserved.")

