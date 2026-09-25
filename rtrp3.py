# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Step 2: Load dataset
data = pd.read_csv("store_sales (1).csv")

print("Columns:", data.columns)

# Step 3: Convert categorical data to numeric
data['Gender'] = data['Gender'].astype('category').cat.codes
data['Category'] = data['Category'].astype('category').cat.codes
data['ItemPurchased'] = data['ItemPurchased'].astype('category').cat.codes
data['Season'] = data['Season'].astype('category').cat.codes
data['PaymentMethod'] = data['PaymentMethod'].astype('category').cat.codes

# Step 4: Feature Engineering (NEW 🔥)
data['DiscountAmount'] = data['Amount'] * data['DiscountApplied(%)'] / 100
data['SpendingScore'] = data['PreviousPurchases'] * data['Amount']

# Step 5: Define Features and Target
X = data[['Age', 'Gender', 'Category', 'ItemPurchased',
          'Season', 'PaymentMethod', 'ItemRating',
          'DiscountApplied(%)', 'PreviousPurchases',
          'DiscountAmount', 'SpendingScore']]

Y = data['Amount']

# Step 6: Scaling (important)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Step 7: Train-Test Split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# ===============================
# 🔹 LINEAR REGRESSION
# ===============================
lr_model = LinearRegression()
lr_model.fit(X_train, Y_train)
lr_pred = lr_model.predict(X_test)



# ===============================
# 📊 EVALUATION
# ===============================
print("\n🔹 Linear Regression Performance")
print("R2 Score:", r2_score(Y_test, lr_pred))
print("MAE:", mean_absolute_error(Y_test, lr_pred))
print("RMSE:", np.sqrt(mean_squared_error(Y_test, lr_pred)))



# ===============================
# 📈 GRAPH (Actual vs Predicted)
# ===============================
plt.figure()

plt.plot(Y_test.values[:100], label="Actual")
plt.plot(lr_pred[:100], label="Linear Regression")

plt.legend()
plt.title("Model Comparison: Actual vs Predicted")
plt.xlabel("Samples")
plt.ylabel("Amount")
plt.show()

# ===============================
# 🔮 FORECASTING (NEW DATA)
# ===============================
# Input format:
# Age, Gender, Category, ItemPurchased, Season,
# PaymentMethod, ItemRating, Discount%, PreviousPurchases, DiscountAmount, SpendingScore

new_data = np.array([[25, 1, 2, 3, 1, 2, 4, 10, 5,
                      25*10/100, 5*25]])

new_data = scaler.transform(new_data)

lr_forecast = lr_model.predict(new_data)



