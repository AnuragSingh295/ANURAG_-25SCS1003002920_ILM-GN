# ANURAG_ 25SCS1003002920_ILM GN
My college acdemic projects and coding work

# Climate Change & Traditional Farming – Working Project Code
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ----------------------------------------------------------
# 1. SYNTHETIC DATA GENERATION
# ----------------------------------------------------------
np.random.seed(42)
n = 200

temperature = np.random.normal(26, 2, n)             # °C
rainfall = np.random.normal(850, 100, n)             # mm
soil_ph = np.random.normal(6.5, 0.4, n)
traditional_practice = np.random.choice([0, 1], n)    # 1 = uses traditional methods

# Crop yield influenced by climate
yield_value = (
    2000 
    - (temperature - 25) * 25
    + (rainfall - 800) * 0.6
    + (soil_ph - 6.5) * 40
    - traditional_practice * 40         # Slightly lower yield using old methods
    + np.random.normal(0, 80, n)        # Noise
)

df = pd.DataFrame({
    "Temperature": temperature,
    "Rainfall": rainfall,
    "Soil_pH": soil_ph,
    "Traditional": traditional_practice,
    "Yield": yield_value
})

print("\n=== SAMPLE DATA ===")
print(df.head())

# ----------------------------------------------------------
# 2. TRAIN/TEST SPLIT
# ----------------------------------------------------------
X = df[["Temperature", "Rainfall", "Soil_pH", "Traditional"]]
y = df["Yield"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------------------------------------
# 3. MODEL TRAINING
# ----------------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# ----------------------------------------------------------
# 4. PREDICTION & EVALUATION
# ----------------------------------------------------------
y_pred = model.predict(X_test)

rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print("\n=== MODEL RESULTS ===")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.3f}")

# ----------------------------------------------------------
# 5. SIMPLE GRAPH (Yield vs Rainfall)
# ----------------------------------------------------------
plt.scatter(df["Rainfall"], df["Yield"])
plt.xlabel("Rainfall (mm)")
plt.ylabel("Crop Yield (kg/ha)")
plt.title("Yield vs Rainfall")
plt.show()