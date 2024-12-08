import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import joblib

# Load training and testing datasets
train_data = pd.read_csv('./C964/train.csv')
test_data = pd.read_csv('./C964/test.csv')

# Display data previews
st.title("Demand Forecasting Application")

st.subheader("Training Data Preview")
st.write(train_data.head())

st.subheader("Testing Data Preview")
st.write(test_data.head())

# Define target and features
target = st.selectbox("Select the target column (from training data)", train_data.columns)
features = st.multiselect("Select feature columns (from training data)", [col for col in train_data.columns if col != target])

if target and features:
    X_train = train_data[features]
    y_train = train_data[target]

    X_test = test_data[features]
    y_test = test_data[target]

# Train the Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Save the trained model
joblib.dump(rf_model, "rf_model.pkl")
st.success("Model trained and saved!")

# Generate predictions
y_pred = rf_model.predict(X_test)

# Evaluate the model
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

# Display evaluation metrics
st.subheader("Model Accuracy Metrics")
st.write(f"R-squared: {r2:.4f}")
st.write(f"Mean Absolute Error (MAE): {mae:.4f}")
st.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--r')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted")
st.pyplot(plt)

st.subheader("Make Predictions")

# Create input fields for user data
input_data = {}
for feature in features:
    input_data[feature] = st.number_input(f"Enter {feature} value", value=0.0)

# Make predictions
if st.button("Predict"):
    input_df = pd.DataFrame([input_data])
    prediction = rf_model.predict(input_df)
    st.write(f"Predicted {target}: {prediction[0]:.2f}")
