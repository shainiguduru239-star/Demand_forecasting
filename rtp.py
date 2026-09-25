# 1. Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 2. Load dataset
data = pd.read_csv("store_sales (1).csv")

# 3. Display first rows
print(data.head())

# 4. Select features (independent variables)
X = data[['Age','ItemRating','DiscountApplied(%)','PreviousPurchases']]

# 5. Target variable (Demand / Sales)
y = data['Amount']

# 6. Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 7. Create model
model = LinearRegression()

# 8. Train model
model.fit(X_train, y_train)

# 9. Predict demand
y_pred = model.predict(X_test)

# 10. Evaluation
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

# 11. Graph: Actual vs Predicted Demand
plt.figure()

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Demand")
plt.ylabel("Predicted Demand")
plt.title("Actual vs Predicted Demand (Linear Regression)")

plt.show()