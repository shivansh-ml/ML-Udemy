import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv('Salary_Data.csv')

x=df.iloc[:,:-1].values
y=df.iloc[:,-1].values

x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

print(regressor.coef_)
print(regressor.intercept_)

print(regressor.predict([[2]]))

'''
Eqn--y=b0+b1x

Assumptions of Linear Regression
1)Linearity--(Linear relation between x and y)
2)Homoscedasticity--(Equal Variance) //We dont want to see a cone type shape increasing or decreasing cone
It would mean that variance is dependent on independent variable
3)Multivariate Normality--(Normality of error distribution) //Intiutively if you look along the line of linear regression,
We want to see normal distribution of a data point
4)Independence of Observations--(No auto-correlations) //We do not want to see any kind of pattern in our data ,
which indicate that rows are independent of each other
5)Lack of Multicollinearity--(Predictors are not correlated with each other) //If they are collinear with each other,
the coefficent of the linear regression model is unreliable
This means they carry overlapping information about the outcome variable, which can cause problems for interpreting the model.
6)Outlier check--(not an assumption but extra) //If outlier is significantly affecting the linear regression line,
then we either remove outlier or keeping them depends on business knowledge 
'''