# -------------------------------
# 1. Import Libraries
# -------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------------------
# 2. Load Dataset
# -------------------------------
df = pd.read_csv("store_sales (1).csv")

print("Dataset Preview:")
print(df.head())

# -------------------------------
# 3. Data Preprocessing
# -------------------------------
# Select only numeric columns
df = df.select_dtypes(include=[np.number])

# Remove missing values
df = df.dropna()

# -------------------------------
# 4. Features and Target
# -------------------------------
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# -------------------------------
# 5. Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# 🔵 LINEAR REGRESSION MODEL
# ===============================
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

y_pred = lr_model.predict(X_test)

# -------------------------------
# Evaluation
# -------------------------------
print("\n--- Linear Regression ---")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Train vs Test Score
print("Train Score:", lr_model.score(X_train, y_train))
print("Test Score:", lr_model.score(X_test, y_test))

# -------------------------------
# Feature Importance
# -------------------------------
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": lr_model.coef_
})

print("\nFeature Importance:")
print(importance)

# ===============================
# 🟢 DECISION TREE MODEL
# ===============================
dt_model = DecisionTreeRegressor()
dt_model.fit(X_train, y_train)

y_pred_dt = dt_model.predict(X_test)

print("\n--- Decision Tree ---")
print("MAE:", mean_absolute_error(y_test, y_pred_dt))
print("MSE:", mean_squared_error(y_test, y_pred_dt))
print("R2 Score:", r2_score(y_test, y_pred_dt))



# ===============================
# 📈 CLEAN LINE CHART
# ===============================
results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

# Sort values for clean graph
results = results.sort_values(by="Actual").reset_index(drop=True)

# Take only first 100 points
results = results.head(100)

plt.figure()

plt.plot(results["Actual"], label="Actual Sales")
plt.plot(results["Predicted"], label="Predicted Sales")

plt.title("Actual vs Predicted Sales (Clean Line Chart)")
plt.xlabel("Samples (Sorted)")
plt.ylabel("Sales")

plt.legend()
plt.grid(True)

plt.show()