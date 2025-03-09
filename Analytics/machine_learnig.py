import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load preprocessed dataset
df = pd.read_csv('generated_data.csv')

# Define features and target
X = df.drop(columns=['Carbon Deposit Level'])
y = df['Carbon Deposit Level']

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=100, random_state=42)
}

# Train and evaluate models
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    results[name] = {'MSE': mse, 'RMSE': rmse, 'R²': r2}

# Display results
results_df = pd.DataFrame(results).T
print("\nModel Performance:")
print(results_df)

# Visualize model performance
plt.figure(figsize=(8, 5))
results_df['RMSE'].plot(kind='bar', color=['blue', 'green', 'red'])
plt.title('Model Comparison - RMSE')
plt.ylabel('Root Mean Squared Error')
plt.xticks(rotation=0)
plt.show()

# Save results
results_df.to_csv('model_performance.csv')
print("\nBest model based on RMSE:", results_df['RMSE'].idxmin())
