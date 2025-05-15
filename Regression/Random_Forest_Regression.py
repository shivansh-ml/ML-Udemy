'''
Random Forest is an ensemble learning method, specifically a type of ensemble based on Decision Trees.
Random Forest is a combination of multiple decision trees that are trained on different subsets of the data.
The idea behind Random Forest is that by combining the predictions of multiple decision trees, we can improve the accuracy of prediction

Ensemble Learning means combining the predictions of multiple models (either different models or the same model trained multiple times) to make a final prediction that is usually more accurate, robust, and stable.
There are different types of ensemble learning:
Bagging (Bootstrap Aggregating) – e.g., Random Forest
Boosting – e.g., Gradient Boosting, AdaBoost, XGBoost
Stacking – combining different types of models using a meta-model

Step - 1: Pick at random K data points from the Training set
Step - 2: Build the Decision Tree associated to these K data points
Step - 3: Choose the number Ntree of trees you want to build and repeat steps 1&2
Step - 4:For a new data point, make each one of your Ntree predict the value of Y to
for the data point in question, and assign the new data point the average across all of 
the predicted Y values

'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=10, random_state=0)
regressor.fit(X, y)

print(regressor.predict([[6.5]]))

X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()