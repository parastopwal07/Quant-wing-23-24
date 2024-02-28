import numpy as np
import pandas as pd

# Step 1: Gather historical data (you can replace this with real-world data)
# Example data for demonstration
data = {
    'Stock1': [0.05, 0.04, 0.03, 0.02, 0.01],
    'Stock2': [0.03, 0.035, 0.025, 0.015, 0.02],
    'Stock3': [0.04, 0.045, 0.035, 0.025, 0.03],
    'Stock4': [0.02, 0.025, 0.015, 0.005, 0.01],
    'Stock5': [0.035, 0.04, 0.03, 0.025, 0.03]
}
df = pd.DataFrame(data)

# Step 2: Calculate mean returns and covariance matrix
mean_returns = df.mean()
cov_matrix = df.cov()

# Step 3 & 4: Generate random portfolios and calculate returns and standard deviation
num_portfolios = 10000
results = np.zeros((3, num_portfolios))  # Rows: return, stdev, weight

for i in range(num_portfolios):
    weights = np.random.random(5)
    weights /= np.sum(weights)
    
    portfolio_return = np.sum(mean_returns * weights)
    portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    results[0, i] = portfolio_return
    results[1, i] = portfolio_stddev
    results[2, i] = portfolio_return / portfolio_stddev  # Sharpe ratio

# Step 5: Determine Value at Risk (VaR) for each portfolio
confidence_level = 0.95
sorted_returns = np.sort(results[0])
var_index = int(len(sorted_returns) * (1 - confidence_level))
value_at_risk = sorted_returns[var_index]

# Step 6: Calculate Expected Shortfall (ES) for each portfolio
expected_shortfall = np.mean(sorted_returns[sorted_returns < value_at_risk])

# Step 7: Evaluate the performance
print("Value at Risk (VaR) at 95% confidence level:", value_at_risk)
print("Expected Shortfall (ES) at 95% confidence level:", expected_shortfall)
