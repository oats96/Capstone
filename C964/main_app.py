# Importing everything
import streamlit
import pandas
import numpy
import joblib
import matplotlib.pyplot
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Function to clean data
def clean_data(data, features, target=None, is_training=True):
    if is_training:
        if target in data.columns:
            data = data.dropna(subset=[target])  # Remove rows with missing target
            median_value = data[target].median()
            data[target] = data[target].fillna(median_value)  # Fill missing target with median
        else:
            print("Target column is missing")  # Debugging

    for feature in features:
        if feature in data.columns:
            if data[feature].dtype == "object":
                mode = data[feature].mode()[0]
                data[feature] = data[feature].fillna(mode)
            else:
                median = data[feature].median()
                data[feature] = data[feature].fillna(median)
    return data

# Load datasets
train_data = pandas.read_csv("./C964/data/train.csv")
test_data = pandas.read_csv("./C964/data/test.csv")

# Extract date features from week column
train_data["week"] = pandas.to_datetime(train_data["week"], format="%y/%m/%d")
test_data["week"] = pandas.to_datetime(test_data["week"], format="%y/%m/%d")

train_data["year"] = train_data["week"].dt.year
train_data["month"] = train_data["week"].dt.month
train_data["day"] = train_data["week"].dt.day
test_data["year"] = test_data["week"].dt.year
test_data["month"] = test_data["week"].dt.month
test_data["day"] = test_data["week"].dt.day

# Drop original week column
train_data = train_data.drop(columns=["week"])
test_data = test_data.drop(columns=["week"])

# Set features and target
target = "units_sold"
features = ["year", "month", "day", "store_id", "sku_id", "base_price", "is_featured_sku", "is_display_sku"]

# Clean data
train_data = clean_data(train_data, features, target=target, is_training=True)
test_data = clean_data(test_data, features, is_training=False)

X_train = train_data[features]
y_train = train_data[target]
X_test = test_data[features]

# Scale numerical data
scaler = StandardScaler()
train_data["base_price"] = scaler.fit_transform(train_data[["base_price"]])
test_data["base_price"] = scaler.transform(test_data[["base_price"]])

# Encode categorical data
encoder = LabelEncoder()
train_data["store_id"] = encoder.fit_transform(train_data["store_id"])
test_data["store_id"] = encoder.transform(test_data["store_id"])

# Train model
rf_model = RandomForestRegressor(n_estimators=50, random_state=42)
rf_model.fit(X_train, y_train)

# Save model
joblib.dump(rf_model, "rf_model.pkl")

# Make predictions
y_pred = rf_model.predict(X_test)

# Calculate metrics
r2 = r2_score(y_train, rf_model.predict(X_train))
mae = mean_absolute_error(y_train, rf_model.predict(X_train))
rmse = numpy.sqrt(mean_squared_error(y_train, rf_model.predict(X_train)))

# Add predictions to test data
test_data["predicted_units_sold"] = y_pred

# Streamlit app
streamlit.title("Demand Forecasting Application")

streamlit.subheader("Training Data Sample")
streamlit.write(train_data.head())

streamlit.subheader("Predictions on Test Data")
streamlit.write(test_data[["sku_id", "predicted_units_sold"]].head())

streamlit.subheader("Model Metrics")
streamlit.write("R2 Score: " + str(r2))
streamlit.write("Mean Absolute Error: " + str(mae))
streamlit.write("Root Mean Squared Error: " + str(rmse))

# Visualization 1: Feature Importance
streamlit.subheader("Feature Importance")
matplotlib.pyplot.barh(features, rf_model.feature_importances_, color="green")
matplotlib.pyplot.title("Feature Importance")
matplotlib.pyplot.xlabel("Importance")
matplotlib.pyplot.ylabel("Features")
streamlit.pyplot(matplotlib.pyplot)

# Visualization 2: Scatter Plot
streamlit.subheader("Scatter Plot of Predicted Units Sold")
matplotlib.pyplot.scatter(test_data["sku_id"], test_data["predicted_units_sold"], color="blue", alpha=0.7)
matplotlib.pyplot.title("Predicted Units Sold by SKU")
matplotlib.pyplot.xlabel("SKU ID")
matplotlib.pyplot.ylabel("Predicted Units")
matplotlib.pyplot.xticks(rotation=90)
streamlit.pyplot(matplotlib.pyplot)

# Visualization 3: Bar Chart
streamlit.subheader("Predicted Units Sold (Top SKUs)")
top_n = streamlit.slider("Number of SKUs to Display", 5, 20, 10)
sku_sales = test_data.groupby("sku_id")["predicted_units_sold"].sum().sort_values(ascending=False).head(top_n)
matplotlib.pyplot.bar(sku_sales.index, sku_sales.values, color="orange")
matplotlib.pyplot.title("Top " + str(top_n) + " SKUs by Predicted Sales")
matplotlib.pyplot.xlabel("SKU")
matplotlib.pyplot.ylabel("Predicted Units Sold")
matplotlib.pyplot.xticks(rotation=45)
streamlit.pyplot(matplotlib.pyplot)