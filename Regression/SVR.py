'''
SVR tries to:
Fit the best line within a margin of tolerance, called epsilon (ε).
Ignore small errors (those within the margin), and
Focus only on points outside this margin to make the model better.

'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

y = y.reshape(len(y),1)

print(x)
print(y)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x = sc_x.fit_transform(x)

sc_y = StandardScaler()
y = sc_y.fit_transform(y)

from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(x, y)

#print(sc_y.inverse_transform(regressor.predict(sc_x.transform([[6.5]])).reshape(-1,1)))
#[[]] because regresor expects any input as 2d array
## Visualising the SVR results
plt.scatter(sc_x.inverse_transform(x), sc_y.inverse_transform(y), color = 'red')
plt.plot(sc_x.inverse_transform(x), sc_y.inverse_transform(regressor.predict(x).reshape(-1,1)), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

'''
[[6.5]] → yes, this is 2D → shape (1,1)
sc_x.transform([[6.5]]) → still shape (1,1) (scaled)
regressor.predict(...) → SVR always returns a 1D NumPy array of predictions
Shape: (1,) (just one number inside an array, like [0.73])
reshape(-1,1) → converts (1,) → (1,1) (column vector), because the scaler sc_y.inverse_transform() expects 2D input
'''

## Visualising the SVR results (for higher resolution and smoother curve)
X_grid = np.arange(min(sc_x.inverse_transform(x)), max(sc_x.inverse_transform(x)), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(sc_x.inverse_transform(x), sc_y.inverse_transform(y), color = 'red')
plt.plot(X_grid, sc_y.inverse_transform(regressor.predict(sc_x.transform(X_grid)).reshape(-1,1)), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()