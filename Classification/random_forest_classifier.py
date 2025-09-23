import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Social_Network_Ads.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from sklearn.ensemble import RandomForestClassifier
classifer = RandomForestClassifier(n_estimators=1000, random_state=10, criterion='entropy')
classifer.fit(x_train,y_train)

y_pred= classifer.predict(x_test)
# Reshape to column vectors
y_pred_col = y_pred.reshape(len(y_pred), 1)
y_test_col = y_test.reshape(len(y_test), 1)

# Concatenate side by side
comparison = np.concatenate((y_pred_col, y_test_col), axis=1)
print(comparison)

from sklearn.metrics import confusion_matrix, accuracy_score
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

from matplotlib.colors import ListedColormap

# Function to plot decision boundary
def plot_RF(X_set, y_set, title):
    X1, X2 = np.meshgrid(
        np.arange(X_set[:, 0].min() - 1, X_set[:, 0].max() + 1, 0.01),
        np.arange(X_set[:, 1].min() - 1, X_set[:, 1].max() + 1, 0.01)
    )
    Z = classifer.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape)
    plt.contourf(X1, X2, Z, alpha=0.75, cmap=ListedColormap(['#FA8072', '#1E90FF']))
    colors = ['#FA8072', '#1E90FF']
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], color=colors[i], label=j)
    plt.title(title)
    plt.xlabel('Age (scaled)')
    plt.ylabel('Estimated Salary (scaled)')
    plt.legend()
    plt.show()

# Plot decision boundaries
plot_RF(x_train, y_train, 'Random Forest (Training set)')
plot_RF(x_test, y_test, 'Random Forest (Test set)')