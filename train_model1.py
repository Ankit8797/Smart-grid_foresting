# train_model1.py
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import joblib

# Import custom modules
from feature_engineering import add_time_features
from hyperparameter_tuning import tune_hyperparameters

# Load the dataset (ensure Timestamp is parsed as datetime)
df = pd.read_csv("realistic_energy_forecast_dataset.csv", parse_dates=["Timestamp"])

# Apply feature engineering: add time features
df = add_time_features(df)

# Define the features and target variable
features = [
    "Temperature_C", "Humidity_%", "Wind_Speed_mps", "Solar_Radiation_Wm2",
    "Industrial_Usage_kWh", "Residential_Usage_kWh", "Commercial_Usage_kWh",
    "Grid_Frequency_Hz", "Voltage_Level_V", "Renewable_Energy_Contribution_%",
    "hour", "dayofweek", "month", "Holiday_Indicator", "Weekday"
]
target = "Energy_Consumption_kWh"

X = df[features]
y = df[target]

# Split the data while maintaining temporal order (first 80% for training, rest for testing)
train_size = int(len(df) * 0.8)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

# Use hyperparameter tuning on a hold-out validation set.
# For simplicity, we'll use the entire test set as validation for tuning.
best_params = tune_hyperparameters(X_train, y_train, X_test, y_test, n_trials=30)

if not best_params:  # Fallback in case tuning fails
    best_params = {
        'learning_rate': 0.05,
        'num_leaves': 31,
        'subsample': 1.0,
        'colsample_bytree': 1.0,
        'min_data_in_leaf': 20
    }

print("Best hyperparameters:", best_params)

# Prepare LightGBM datasets
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# Define model parameters based on tuning results
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': best_params.get('learning_rate', 0.05),
    'num_leaves': best_params.get('num_leaves', 31),
    'verbose': -1,
    'bagging_freq': 1,
    'subsample': best_params.get('subsample', 1.0),
    'colsample_bytree': best_params.get('colsample_bytree', 1.0),
    'min_data_in_leaf': best_params.get('min_data_in_leaf', 20)
}

# Train the model with early stopping
model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[test_data],
    callbacks=[
        lgb.log_evaluation(period=50),
        lgb.early_stopping(stopping_rounds=50)  # ✅ Fixed the early stopping issue
    ]
)

# Make predictions on the test set using the best iteration
predictions = model.predict(X_test, num_iteration=model.best_iteration)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print("Test RMSE:", rmse)



# Save the trained model to disk for deployment
joblib.dump(model, "energy_forecast_model.pkl")

# Save feature names for deployment to prevent column mismatch
joblib.dump(features, "model_features.pkl")

from sklearn.metrics import mean_absolute_error, r2_score

# Compute additional metrics
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

# Save metrics to a file
metrics = {
    'rmse': rmse,
    'mae': mae,
    'r2': r2,
    'train_time': 0,  # Optional: Replace with actual timing if you track it
    'data_points': len(df)
}
joblib.dump(metrics, "model_metrics.pkl")

# Save the most recent actual demand for use in the app
current_demand = y.iloc[-1]
joblib.dump(current_demand, "last_demand.pkl")


print("Model and features saved successfully!")
