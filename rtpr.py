# -------------------------------
# 1. Import Libraries
# -------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -------------------------------
# 2. Load Dataset
# -------------------------------
df = pd.read_csv("store_sales (1).csv")

print(df.head())   # check data

# -------------------------------
# 3. Data Preprocessing
# -------------------------------
# Select only numeric columns
df = df.select_dtypes(include=[np.number])

# Handle missing values
df = df.dropna()

# -------------------------------
# 4. Features and Target
# -------------------------------
# Assume last column is target (Sales/Demand)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# -------------------------------
# 5. Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 6. Train Model
# -------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -------------------------------
# 7. Predictions
# -------------------------------
y_pred = model.predict(X_test)

# -------------------------------
# 8. Evaluation
# -------------------------------
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("MAE:", mae)
print("MSE:", mse)

# -------------------------------
# 9. LINE CHART (IMPORTANT)
# -------------------------------
# Sort values for smooth graph
sorted_indices = np.argsort(y_test)

y_test_sorted = y_test.iloc[sorted_indices]
y_pred_sorted = y_pred[sorted_indices]
n=10

plt.figure()

plt.plot(y_test_sorted.values, label="Actual Sales")
plt.plot(y_pred_sorted, label="Predicted Sales")

plt.title("Actual vs Predicted Sales (Line Chart)")
plt.xlabel("Samples")
plt.ylabel("Sales")

plt.legend()
plt.show()