# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load dataset
df = pd.read_csv("shopping_trends.csv")   # change file name if needed

# Select useful columns (numerical only for simplicity)
X = df[['Age', 'Amount', 'ItemRating', 'DiscountApplied(%)', 'PreviousPurchases']]
y = df['Amount']   # predicting demand (you can change target if needed)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("MAE:", mae)
print("MSE:", mse)

# -------------------------------
# 📈 LINE CHART (Actual vs Predicted)
# -------------------------------

# Sort values for smooth line graph
sorted_indices = np.argsort(y_test)
y_test_sorted = y_test.iloc[sorted_indices]
y_pred_sorted = y_pred[sorted_indices]

plt.figure()

plt.plot(y_test_sorted.values, label="Actual Demand")
plt.plot(y_pred_sorted, label="Predicted Demand")

plt.title("Actual vs Predicted Demand (Line Chart)")
plt.xlabel("Samples")
plt.ylabel("Demand")

plt.legend()
plt.show()