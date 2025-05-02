'''
Polynomial Regression (Degree 3):
y = b0 + b1*x + b2*x^2 + b3*x^3
Example: y = 0.5 + 1*x + 0.2*x^2 + 0.05*x^3
It is also a type of non-linear regression model.
It is called polynomial-Linear regression.Because it's still linear in terms of the coefficients, even though the data relationship is non-linear.
The model is linear in its parameters (the b's), which is what matters when fitting the model using methods like Ordinary Least Squares (OLS).
It is a special case of Multi-Linear Regression

'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x, y)
#we didn't split the data set into a training set and a test set,because we want to leverage the maximum data in order to train our model.

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(x)
lin_reg = LinearRegression()
lin_reg.fit(X_poly,y)

#Visualizing the Linear Regression results
plt.scatter(x, y, color = 'red')
plt.plot(x, regressor.predict(x), color='blue')
plt.title('Truth or Bluff(Linear Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#Visualising the Polynomial Regeression result
plt.scatter(x, y, color = 'red')
plt.plot(x, lin_reg.predict(X_poly), color='blue')
plt.title('Truth or Bluff(Polynomial Linear Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# ##Visualising the Polynomial Regeression result(for higher resolution and smoother curve)
# X_grid = np.arange(min(x), max(x), 0.1)#Creates a fine-grained set of x values from the minimum to maximum level (e.g., 1.0, 1.1, 1.2, ..., 10.0).
# #Makes the plot look smoother by computing predictions at many intermediate points.
# X_grid = X_grid.reshape((len(X_grid), 1))#Converts the array into a 2D shape required by scikit-learn models.
# plt.scatter(x, y, color = 'red')
# plt.plot(X_grid, lin_reg.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
# #Predicts salary for every 0.1 step in X, giving a much smoother curve.
# plt.title('Truth or Bluff (Polynomial Regression)')
# plt.xlabel('Position level')
# plt.ylabel('Salary')
# plt.show()

print(regressor.predict([[6.5]]))#1st [] corresponds to rows 2nd [] corresponds to columns
print(lin_reg.predict(poly_reg.fit_transform([[6.5]])))