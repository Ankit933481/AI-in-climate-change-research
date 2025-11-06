# main.py
# AI-Based Climate Prediction and Data Completion System
# Author: Ankit Kumar | Chandigarh University
# Description: Demonstrating how AI overcomes missing data and improves climate prediction accuracy.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer

# -------------------------------
# Step 1: Create or Load Dataset
# -------------------------------
# Simulated climate dataset (Temperature, Humidity, Rainfall, SolarRadiation)
data = {
    'Temperature': [24, 26, np.nan, 30, 31, 29, np.nan, 33, 35, 32, 28, 27, 29, 30, np.nan],
    'Humidity':    [65, 62, 68, np.nan, 55, 59, 63, 60, 58, np.nan, 61, 66, 67, 64, 62],
    'Rainfall':    [110, 95, 120, 130, 80, np.nan, 90, 105, 115, 100, 85, np.nan, 125, 135, 98],
    'SolarRadiation': [420, 460, 440, 480, 500, 470, np.nan, 510, 520, 490, 450, 430, np.nan, 515, 475],
    'TemperatureNextDay': [25, 27, 28, 31, 32, 30, 32, 34, 36, 33, 29, 28, 30, 31, 32]
}

df = pd.DataFrame(data)
print("=== Original Climate Data (with Missing Values) ===")
print(df)

# -----------------------------------------
# Step 2: Handle Missing Data using Imputer
# -----------------------------------------
imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

print("\n=== Data After Imputation (Missing Values Filled by AI) ===")
print(df_imputed)

# -----------------------------------------
# Step 3: Train-Test Split
# -----------------------------------------
X = df_imputed[['Temperature', 'Humidity', 'Rainfall', 'SolarRadiation']]
y = df_imputed['TemperatureNextDay']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------------------
# Step 4: Train Random Forest Model
# -----------------------------------------
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -----------------------------------------
# Step 5: Make Predictions
# -----------------------------------------
y_pred = model.predict(X_test)

# -----------------------------------------
# Step 6: Evaluate Model
# -----------------------------------------
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n=== Model Evaluation ===")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R² Score: {r2:.2f}")

# -----------------------------------------
# Step 7: Predict Future Temperature (Demo)
# -----------------------------------------
new_data = pd.DataFrame({
    'Temperature': [31],
    'Humidity': [58],
    'Rainfall': [100],
    'SolarRadiation': [510]
})

prediction = model.predict(new_data)
print("\nPredicted Next-Day Temperature (°C):", prediction[0])
