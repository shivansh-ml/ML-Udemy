import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# Load dataset
dataset = pd.read_csv("CollegePlacement.csv")

# Encode target
le = LabelEncoder()
y = le.fit_transform(dataset.iloc[:, -1].values)

# Features
x = dataset.iloc[:, 1:-1].values

# Encode categorical feature(s) (example: column 4)
x[:, 4] = LabelEncoder().fit_transform(x[:, 4])

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# Scale numeric features (example: first 2 cols)
sc = StandardScaler()
x_train[:, 0:2] = sc.fit_transform(x_train[:, 0:2])
x_test[:, 0:2] = sc.transform(x_test[:, 0:2])

# Base learners
estimators = [
    ('knn', KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)),
    ('nb', GaussianNB()),
    ('dt', DecisionTreeClassifier(criterion='entropy', random_state=0)),
    ('svm', SVC(kernel='rbf', probability=True, random_state=0)),  # kernel SVM
    ('log', LogisticRegression(max_iter=200, random_state=0))
]

# Meta learner (Random Forest)
stacking_clf = StackingClassifier(
    estimators=estimators,
    final_estimator=RandomForestClassifier(n_estimators=10, random_state=0),
    cv=5
)

# Train stacking model
stacking_clf.fit(x_train, y_train)

# Predictions
y_pred = stacking_clf.predict(x_test)

# Evaluation
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))