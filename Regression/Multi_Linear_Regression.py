'''
Dummy-Variable Trap
We have to omit one dummy variable because the model cannot distinguish the difference between the effects of D1 and D2 when D2= 1 - D1. 
This can cause multicollinearity, meaning the model has a hard time predicting certain variables. 
5 method of buiding a model:
1)All-in
2)Backward Elimination----------------Stepwise Regression
3)Forward Elimination-----------------Stepwise Regression
4)Bidirectional Elimination-----------Stepwise Regression
5)Score Comparison

p-value means how likely the variable is not useful
Interpretation of p-value:
p < 0.05 → The feature is statistically significant → keep it.
p > 0.05 → The feature is not significant → consider removing i

All-in:throw in all the variable(need to have prior knowledge or if it is necessary or preparing for backward elimination)

Backward Elimination:
Step-1:Select a significance level to stay in the model(e.g. SL or alpha=0.05)
Step-2:Fit the full model with all posssible predictors
Step-3:Consider the predictor with the highest Probability-value.
       If P>SL, go to step-4,otherwise go to FIN
Step-4:Remove the predictor
Step-5:Fit model without the variable
       After Step-5 we go to Step-3
We do that until we come to the point where the highest P-valye is less than SL
FIN:MODEL IS READY
Eg-You start with all your friends in a team and slowly remove the ones who are not helping much.


Forward Selection:
Step-1:Select a significance level to stay in the model(e.g. SL or alpha=0.05)
Step-2:Fit all simple regression models Y-Xn .Select the one with the lowest P-value
       We take all the independent variable and apply linear regression to each of them wrt dependent variable
Step-3:Keep this variable and fit all possible models with one extra predictor added to the one(s) you already have
Step-4:Consider the predictor with lowest P-value.If P<Sl go to step-3, otherwise go to FIN
FIN:KEEP THE PREVIOUS MODEL
Eg-You start alone and slowly add teammates who really help


Bidirectional Elimination:
Step-1:Select a significance level to stay in the model(e.g. SLENTER = 0.05 AND SLSTAY=0.05)
Step-2:Perform the next step of Forward Elimination(new variable must have P<SLENETR to enetr)
Step-2:Perform the next step of Backward Elimination(old variable must have P<SLSTAY to stay)
       We move from Step-2 to Step-3
Step-4:No new variables can enter and no old variables can exit 
FIN: YOUR MODEL IS READY
Eg-You’re building a team — invite good friends, but remove old ones if they stop contributing.

All possible model:
Step-1:Select a criterion of goodness of it(e.g. - Akaike criterion)
Step-2:Construct All possible Regression Models.2^N - 1 total combiantion
Step-3:Select the one with best criterion
Eg-You try every team combination and choose the one that works the best.

There is no need to apply feature scaling
We cannot plot a graph as we have multiple features
'''
'''
ColumnTransformer → a function from sklearn.compose that allows us to apply different transformations to specific columns in a dataset.
'encoder':Just a name for this transformation (you can call it anything).
OneHotEncoder():This tells Python to apply One-Hot Encoding to a column.
This means: "Apply OneHotEncoder to column number 3" (count starts from 0).
“Apply One-Hot Encoding to column 3 of X, keep the rest of the columns as they are, and save the transformed result as a NumPy array in X.”
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

print(X)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)## display decimals with only 2 digits after the decimal point
#set_printoptions(...): This is a NumPy function that changes how numbers are displayed when printed (not how they’re stored or calculated).
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
'''
Multiple Linear Regression:
y = b0 + b1*x1 + b2*x2 + b3*x3 + ... + bn*xn
Example: y = 1.5 + 2*x1 + 0.5*x2 + 3*x3

axis=0 → join vertically (row-wise)
axis=1 → join horizontally (column-wise)
➡️ So axis=1 means: put one array next to the other
.reshape(len(y_pred), 1) turns it into a column vector (with 1 column).
'''