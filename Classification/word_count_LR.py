# Open the file in read mode with utf-8 encoding
text = open("scraped_content.txt", "r", encoding="utf-8", errors="ignore")

# Create an empty dictionary
d = dict()

# Loop through each line of the file
for line in text:
    line = line.strip()
    line = line.lower()
    words = line.split(" ")

    for word in words:
        if word in d:
            d[word] = d[word] + 1
        else:
            d[word] = 1

text.close()  # always good practice to close the file

# Print the contents of dictionary
# for key in list(d.keys()):
#     print(key, ":", d[key])

import pandas as pd
import numpy as np


# Output: dictionary of dictionaries
d_features = {}

for word, count in d.items():
    # compute features
    length = len(word)
    ends_s = 1 if word.endswith("s") else 0
    ends_es = 1 if word.endswith("es") else 0
    ends_ies = 1 if word.endswith("ies") else 0
    is_alpha = word.isalpha()
    if length <= 2 or not is_alpha:
        label = 0   # neither
    elif ends_ies or (ends_s and not word.endswith("ss")):
        label = 2   # plural
    else:
        label = 1   # singular
    # assign features into a nested dictionary
    d_features[word] = {
        "Length": length,
        "ends_s": ends_s,
        "ends_es": ends_es,
        "ends_ies": ends_ies,
        "Count": count,
        # temporary label placeholder (you can set manually later)
        "Label": label
    }

# Print results
# for word, features in d_features.items():
#     print(word, "->", features)

df = pd.DataFrame(d_features).T
print(df)
print(df.head(20))
# Features: all columns except 'Label'
x = df.drop(columns=["Label","Length","Count"]).values    # or df.iloc[:,:-1].values
y = df["Label"].values
# print(x[:20])

# Train-test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Check first 20 rows of x_train
# print("x_train sample:\n", x_train[:80])

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(x_train, y_train)


y_pred = classifier.predict(x_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

from sklearn.metrics import confusion_matrix, accuracy_score
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

word = input("Enter a word: ").lower().strip()
x_new = [[1 if word.endswith("s") else 0,
          1 if word.endswith("es") else 0,
          1 if word.endswith("ies") else 0]]
label_map = {0: "neither", 1: "singular", 2: "plural"}
print(f"The word '{word}' is predicted as: {label_map[classifier.predict(x_new)[0]]}")