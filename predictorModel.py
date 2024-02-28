import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load data
data = pd.read_csv('train.csv')

# Feature Engineering
# Example: Creating a new feature by calculating the moving average
data['Moving_Avg_10'] = data['price'].rolling(window=10).mean()

# Visualize data
plt.figure(figsize=(10, 6))
plt.plot(data['Date'], data['price'], label='Close Price')
plt.plot(data['Date'], data['Moving_Avg_10'], label='Moving Average (10 days)', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Stock Price and Moving Average')
plt.legend()
plt.show()

# Split data into features and target variable
X = data.drop(['Date', 'price'], axis=1)  # Features
y = data['price']  # Target variable

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Regressor
rf_regressor = RandomForestRegressor(random_state=42)
rf_regressor.fit(X_train, y_train)

# Evaluate the model
y_pred = rf_regressor.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'RMSE: {rmse}')

# Feature Importance
feature_importance = rf_regressor.feature_importances_
features = X.columns
sorted_indices = np.argsort(feature_importance)[::-1]
sorted_features = [features[i] for i in sorted_indices]
sorted_importance = feature_importance[sorted_indices]

plt.figure(figsize=(10, 6))
plt.barh(sorted_features, sorted_importance)
plt.xlabel('Importance')
plt.ylabel('Features')
plt.title('Feature Importance')
plt.show()

# Hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=rf_regressor, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print(f'Best Hyperparameters: {best_params}')

# Re-train the model with best hyperparameters
best_rf_regressor = RandomForestRegressor(**best_params, random_state=42)
best_rf_regressor.fit(X_train, y_train)

# Evaluate the model with best hyperparameters
y_pred_best = best_rf_regressor.predict(X_test)
rmse_best = np.sqrt(mean_squared_error(y_test, y_pred_best))
print(f'RMSE with best hyperparameters: {rmse_best}')

# Plot actual vs. predicted prices
plt.figure(figsize=(10, 6))
plt.plot(y_test.index, y_test, label='Actual Prices')
plt.plot(y_test.index, y_pred_best, label='Predicted Prices')
plt.xlabel('Index')
plt.ylabel('Price')
plt.title('Actual vs. Predicted Prices')
plt.legend()
plt.show()
