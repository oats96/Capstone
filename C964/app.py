import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import joblib

# Cleaning function
def clean_data(data, features, target=None, train=True):
    if train and target:
        if target not in data.columns:
            raise ValueError(f"Target column '{target}' is missing from the dataset.")
        if data[target].isnull().sum() > 0:
            data = data.dropna(subset=[target])
        target_median = data[target].median()
        data[target] = data[target].mask(data[target] < 0, target_median)
        upper_limit = data[target].quantile(0.99)
        data[target] = data[target].clip(upper=upper_limit)

    for feature in features:
        if feature in data.columns:
            if data[feature].dtype in ['float64', 'int64']:
                data[feature] = data[feature].fillna(data[feature].median())
            else:
                data[feature] = data[feature].fillna(data[feature].mode()[0])

    if 'base_price' in features and 'base_price' in data.columns:
        median_value = data['base_price'].median()
        data['base_price'] = data['base_price'].mask(data['base_price'] <= 0, median_value)

    return data

# Load datasets
train_data = pd.read_csv('./C964/train.csv')
test_data = pd.read_csv('./C964/test.csv')

# Convert week column to datetime and extract components
train_data['week'] = pd.to_datetime(train_data['week'], format='%y/%m/%d')
test_data['week'] = pd.to_datetime(test_data['week'], format='%y/%m/%d')

train_data['year'] = train_data['week'].dt.year
train_data['month'] = train_data['week'].dt.month
train_data['day'] = train_data['week'].dt.day

test_data['year'] = test_data['week'].dt.year
test_data['month'] = test_data['week'].dt.month
test_data['day'] = test_data['week'].dt.day

# Drop the original week column
train_data.drop(columns=['week'], inplace=True)
test_data.drop(columns=['week'], inplace=True)

# Define target and features
target = 'units_sold'
features = ['year', 'month', 'day', 'store_id', 'sku_id', 'base_price', 'is_featured_sku', 'is_display_sku']

# Clean training data
train_data = clean_data(train_data, features, target=target, train=True)
X_train = train_data[features]
y_train = train_data[target]

# Clean testing data
test_data = clean_data(test_data, features, train=False)
X_test = test_data[features]

# Scale numerical features
scaler = StandardScaler()
train_data['base_price'] = scaler.fit_transform(train_data[['base_price']])
test_data['base_price'] = scaler.transform(test_data[['base_price']])

# Encode categorical features
encoder = LabelEncoder()
train_data['store_id'] = encoder.fit_transform(train_data['store_id'])
test_data['store_id'] = encoder.transform(test_data['store_id'])

# Train the model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Save the model
joblib.dump(rf_model, "rf_model.pkl")

# Generate predictions for test data
y_pred = rf_model.predict(X_test)

# Add predictions to the test dataset
test_data['predicted_units_sold'] = y_pred

# Streamlit App
st.title("Demand Forecasting Application")

st.subheader("Training Data Preview")
st.write(train_data.head())

st.subheader("Testing Data with Predictions")
st.write(test_data[['record_ID', 'predicted_units_sold']].head())

# Interactive graph with a slider
st.subheader("Interactive Visualization of Predictions")
max_points = st.slider("Number of Data Points to Display", 10, len(test_data), 50)

# Interactive scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(test_data['record_ID'][:max_points], test_data['predicted_units_sold'][:max_points], alpha=0.7, color='blue')
plt.xlabel("Record ID")
plt.ylabel("Predicted Units Sold")
plt.title("Predicted Units Sold per Record")
plt.xticks(rotation=45)
plt.grid()
st.pyplot(plt)
