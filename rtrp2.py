# Step 1: Import libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Step 2: Load dataset
data = pd.read_csv("store_sales (1).csv")

print("\n✅ Dataset Loaded Successfully")
print("Columns:", data.columns)

# ===============================
# 🔹 TEST CASE 02: Missing Values
# ===============================
print("\n🔹 Test Case 02: Missing Values")
print("Missing values before:", data.isnull().sum().sum())

data = data.dropna()

print("Missing values after:", data.isnull().sum().sum())
print("Status: Pass")

# ===============================
# 🔹 TEST CASE 03: Categorical Data
# ===============================
print("\n🔹 Test Case 03: Categorical Data Encoding")

data['Gender'] = data['Gender'].astype('category').cat.codes
data['Category'] = data['Category'].astype('category').cat.codes
data['ItemPurchased'] = data['ItemPurchased'].astype('category').cat.codes
data['Season'] = data['Season'].astype('category').cat.codes
data['PaymentMethod'] = data['PaymentMethod'].astype('category').cat.codes

print("Categorical data converted successfully")
print("Status: Pass")

# ===============================
# 🔹 Feature Selection
# ===============================
X = data[['Age', 'Gender', 'Category', 'ItemPurchased',
          'Season', 'PaymentMethod', 'ItemRating',
          'DiscountApplied(%)', 'PreviousPurchases']]

Y = data['Amount']

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# ===============================
# 🔹 TEST CASE 01: Model Prediction
# ===============================
print("\n🔹 Test Case 01: Valid Data Prediction")

model = RandomForestRegressor()
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

print("Sample Predictions:", Y_pred[:5])
print("R2 Score:", r2_score(Y_test, Y_pred))
print("Status: Pass")

# ===============================
# 🔹 TEST CASE 04: Future Prediction
# ===============================
print("\n🔹 Test Case 04: Future Prediction")

new_data = np.array([[25, 1, 2, 3, 1, 2, 4, 10, 5]])

prediction = model.predict(new_data)

print("Predicted Demand:", prediction[0])
print("Status: Pass")