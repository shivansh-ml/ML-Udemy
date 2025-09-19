import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

dataset = pd.read_csv('CollegePlacement.csv')

x = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values
print(x)
print(y)

le=LabelEncoder()
#le → You create a LabelEncoder object.
y =le.fit_transform(y)
print(y)
x[:, 4] = le.fit_transform(x[:, 4])
print(x[:, 4])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = 1)

sc = StandardScaler()
# x_train [:, 0:2]= sc.fit_transform(x_train [:, 0:2])
# x_test [:, 0:2]= sc.transform(x_test[:, 0:2])
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

from sklearn.metrics import confusion_matrix, accuracy_score
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))