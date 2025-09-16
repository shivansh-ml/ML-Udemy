import pandas as pd
import numpy as np

with open("scraped_content.txt", "r", encoding="utf-8", errors="ignore") as f:
    raw_text = f.read()
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform([raw_text])

words = vectorizer.get_feature_names_out()
counts = X.toarray()[0]  # since you only have 1 document
d = {word: int(count) for word, count in zip(words, counts)}
# for word, count in d.items():
#     print(f"{word}: {count}")

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
    elif ends_ies or (ends_s and not word.endswith("ss")) or ends_es:
        label = 2   # plural
    else:
        label = 1   # singular
    
    d_features[word] = {
        "Length": length,
        "ends_s": ends_s,
        "ends_es": ends_es,
        "ends_ies": ends_ies,
        "Count": count,
        "is_alpha": int(is_alpha),
        "Label": label
    }

# Print results
# for word, features in d_features.items():
#     print(word, "->", features)

df = pd.DataFrame(d_features).T

x = df.drop(columns=["Label"]).values
y = df["Label"].values
# print(x[:20])

# Train-test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Check first 20 rows of x_train
# print("x_train sample:\n", x_train[:80])

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0, max_iter=1000)
classifier.fit(x_train, y_train)


y_pred = classifier.predict(x_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

from sklearn.metrics import confusion_matrix, accuracy_score
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

word = input("Enter a word: ").lower().strip()

x_new = [[
    len(word),                          # length
    1 if word.endswith("s") else 0,     # ends_s
    1 if word.endswith("es") else 0,    # ends_es
    1 if word.endswith("ies") else 0,   # ends_ies
    1 if word.isalpha() else 0,
    1                                   # count (always 1 for single word)
]]

label_map = {0: "neither", 1: "singular", 2: "plural"}
pred = classifier.predict(x_new)[0]
print(f"The word '{word}' is predicted as: {label_map[pred]}")