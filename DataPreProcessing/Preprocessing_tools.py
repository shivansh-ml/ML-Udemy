import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
'''
Features/independent variable with which we are going to predict dependent variable
Generally dependent variables are the variables that we are going to predict
'''
dataset = pd.read_csv('Data.csv')
x =  dataset.iloc[:,:-1].values#Independent variable helps to define or predict dependent variable;
#[:] for taking all rows, [,] is to differentiate between rows and columns, [:-1] means we are starting from index 0 going upto -1(last column) i.e. it exludes upper bound i.e. last column
y =  dataset.iloc[:,-1].values#Dependent variable vector

print(x)
print(y)

# Identify missing data (assumes that missing data is represented as NaN)
missing_data = dataset.isnull().sum()
# Print the number of missing entries in each column
print("Missing data: \n", missing_data)

'''
If some data is missing we have to replace it
1)deleting it works for large data set since it doesnt reduce the learning capacity of our model
(Eg-1% is missing for a huge data set)
2)taking average of all columns --- Classic way by using scikit learn -- simpleImputer
3)using median value
4)using most frequent value
5)using a constant
'''

from sklearn.impute import SimpleImputer as SI
imputer = SI(missing_values=np.nan, strategy='mean')
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

print(x)

#Encoding--We are going to convert non-numerical (categorical) data into numbers, because machine learning models work with numbers.
#for independent variable
'''
sklearn.compose → Module in scikit-learn that helps apply transformations to specific columns of data.
ColumnTransformer → A tool that applies transformers (like encoders) only to certain columns.
from sklearn.preprocessing → Importing from the part of scikit-learn that deals with preprocessing (getting data ready).
OneHotEncoder → A tool that turns categories into binary (0/1) vectors.
remainder='passthrough' → Any columns not mentioned should just stay as they are (don't change them).
np.array(...) → Converting the output to a numpy array (for easier handling).
ct.fit_transform(x) fit: Learns how to transform the data based on your input. ➔ transform: Applies the transformation. ➔ Together: it both learns and applies OneHotEncoding to column 0.
'''
 # Identify the categorical data
categorical_columns = dataset.select_dtypes(include=['object']).columns
print("Categorical columns:", categorical_columns)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(),[0])], remainder='passthrough')
x = np.array(ct.fit_transform(x))
print(x)
#for dependent variable

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
#le → You create a LabelEncoder object.
y =le.fit_transform(y)
print(y)

#we apply feature scaling after dataset split into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)
print(X_test)
print(X_train)
print(y_test)
print(y_train)


#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

#fit()-> calculates some statistics from your data.
#It learns the "rules" it needs to later modify the data.
#transform()-> modifies your data using the things learned during fit().
#It changes your dataset according to the rules.

X_train [:, 3:]= sc.fit_transform(X_train [:, 3:])
X_test [:, 3:]= sc.transform(X_test[:, 3:])
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)
print(X_train)
print(X_test)