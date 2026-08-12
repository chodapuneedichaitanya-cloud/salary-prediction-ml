import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Page configuration
st.set_page_config(
    page_title="Salary Prediction",
    page_icon="💰",
    layout="centered"
)

# Title
st.title("💰 Salary Prediction")
st.write("Predict salary based on years of experience using Linear Regression.")

# Load dataset
data = pd.read_csv("salary_Data.csv")

# Input and output
X = data[["YearsExperience"]]
y = data["Salary"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=0
)

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# User input
experience = st.number_input(
    "Enter Years of Experience",
    min_value=0.0,
    max_value=50.0,
    value=5.0,
    step=0.1
)

# Prediction
if st.button("Predict Salary"):
    predicted_salary = model.predict([[experience]])[0]

    st.success(
        f"Predicted Salary: ₹{predicted_salary:,.2f}"
    )

# Show dataset
with st.expander("View Dataset"):
    st.dataframe(data)

# Model information
with st.expander("Model Information"):
    st.write("Model: Linear Regression")
    st.write(f"Training Samples: {len(X_train)}")
    st.write(f"Testing Samples: {len(X_test)}")
    st.write(f"Model Coefficient: {model.coef_[0]:.2f}")
    st.write(f"Model Intercept: {model.intercept_:.2f}")