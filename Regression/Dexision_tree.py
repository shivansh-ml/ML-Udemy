'''
How does it decide where to split?
The algorithm uses a concept called information gain or entropy reduction.
It checks how much "new information" each split gives.
If a split adds very little value, it stops.
You can also set rules like: "Don’t split if fewer than 5% of points are in the segment."

Predicting New Values with the Tree
Once the tree is built, you can use it to predict y values for new data points.
Example: New point with x1 = 30, x2 = 50:
Follow the tree rules (which branches it takes)
It ends up in one of the terminal leaves
The predicted y is the average y value of all training points in that leaf
This is how the tree "regresses" — by assigning average values to regions

This works better than average because If you just took the average y of all points (with no tree), you’d use the same prediction for everything.
With a regression tree:
You get custom averages for each region of the input space (x1, x2).
So the prediction is more local, and hence more accurate.

'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
#random_state=0: Ensures reproducibility. Decision Trees involve some randomness (e.g., in how splits are chosen when multiple splits give the same result).
#Setting random_state=0 means the model will behave the same way every time you run the code (important for testing or tutorials).

regressor.fit(X, y)

regressor.predict([[6.5]])

X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

'''
X_grid = np.arange(min(X), max(X), 0.01)
Purpose: Create a finely spaced grid of values between the minimum and maximum of X.

Why: Decision Tree Regression creates piecewise constant predictions. Using more points gives a smoother, clearer plot of the model’s behavior.

0.01: The step size – smaller values mean more points and a smoother curve.

X_grid = X_grid.reshape((len(X_grid), 1))
Purpose: Reshape the 1D array X_grid into a 2D array with one column.

Why: Scikit-learn models expect a 2D array as input for prediction (n_samples, n_features).

plt.scatter(X, y, color = 'red')
Purpose: Plot the actual data points.

Red dots: Represent the true salary values for each position level.

plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
Purpose: Plot the model’s predictions over the fine grid X_grid.

Blue line: Shows how the Decision Tree regressor predicts salary across position levels.
'''