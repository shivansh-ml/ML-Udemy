
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score

# Load dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Feature scaling
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# SVM classifier
classifier = SVC(kernel='linear', random_state=0)
classifier.fit(x_train, y_train)

# Predict on test set
y_pred = classifier.predict(x_test)

# Confusion matrix and accuracy
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Function to plot decision boundary
def plot_svm(X_set, y_set, title):
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

# Plot decision boundaries
plot_svm(x_train, y_train, 'SVM (Training set)')
plot_svm(x_test, y_test, 'SVM (Test set)')

'''
SVM is a supervised machine learning algorithm used for classification (and regression).
It tries to find the best boundary (hyperplane) that separates classes in the feature space.

2. Key concepts
Hyperplane: A line (2D) or plane (3D) that separates different classes.
Support vectors: The data points closest to the hyperplane. These are crucial because they define the boundary.
Margin: The distance between the hyperplane and the nearest support vectors.
SVM tries to maximize this margin → “maximum margin classifier”.

3. How it works
SVM finds the hyperplane that best separates classes with the largest margin.
For non-linearly separable data, it uses the kernel trick to map data into higher dimensions where it becomes linearly separable.

Common kernels:
linear → straight line boundary
poly → polynomial boundary
rbf → radial (Gaussian) boundary

4. Strengths
Effective in high-dimensional spaces.
Works well when number of features > number of samples.
Can model complex boundaries using kernels.

5. Weaknesses
Not ideal for very large datasets (slow training).
Sensitive to feature scaling → always scale numeric features.
Choice of kernel and parameters (C, gamma) affects performance.
'''