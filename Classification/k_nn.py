import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

dataset=pd.read_csv('./Social_Network_Ads.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier.fit(x_train, y_train)
print(classifier.predict(sc.transform([[30,87000]])))

y_pred = classifier.predict(x_test)
print(y_pred)

from sklearn.metrics import confusion_matrix, accuracy_score
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

def plot_knn(X_set, y_set, title):
    X1, X2 = np.meshgrid(
        np.arange(X_set[:, 0].min() - 1, X_set[:, 0].max() + 1, 0.01),
        np.arange(X_set[:, 1].min() - 1, X_set[:, 1].max() + 1, 0.01)
    )
    Z = classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape)
    plt.contourf(X1, X2, Z, alpha=0.75, cmap=ListedColormap(['#FA8072', '#1E90FF']))
    colors = ['#FA8072', '#1E90FF']
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], color=colors[i], label=j)
    plt.title(title)
    plt.xlabel('Age (scaled)')
    plt.ylabel('Estimated Salary (scaled)')
    plt.legend()
    plt.show()

# Plot training set
plot_knn(x_train, y_train, 'KNN (Training set)')

# Plot test set
plot_knn(x_test, y_test, 'KNN (Test set)')
'''
We take no of neighbours (default is 5)
We will enter the point we want to be predicted then knn will calculate euclidean distance(or some other what you want)
It will find out what is 5 closest neighbours to it and then classify the point as the one label which has maximum no of points near it 

For regression, instead of majority voting, KNN takes the average of the neighbors’ values.
Scaling your features (like you did with StandardScaler) is important because KNN is distance-based, so large-scale differences can skew results.
If you want, I can also draw a small diagram showing KNN in action. It makes the concept super easy to visualize.
'''