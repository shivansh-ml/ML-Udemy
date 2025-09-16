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

def max_consonant_cluster(word):
    max_len = 0
    current_len = 0
    for char in word.lower():
        if char not in "aeiou":
            current_len += 1
            if current_len > max_len:
                max_len = current_len
        else:
            current_len = 0
    return max_len

d_features = {}
for word, count in d.items():
    # compute features
    length = len(word)
    ends_s = 1 if word.endswith("s") else 0
    ends_es = 1 if word.endswith("es") else 0
    ends_ies = 1 if word.endswith("ies") else 0
    is_alpha = word.isalpha()
    has_digit = int(any(c.isdigit() for c in word))
    has_nonalpha = int(not word.isalpha())
    vowel_ratio = sum(c in "aeiou" for c in word.lower()) / max(1, length)
    max_consonant = max_consonant_cluster(word)
    too_short_or_long = int(length < 2 or length > 20)

    if has_digit or has_nonalpha or too_short_or_long or max_consonant > 10 or vowel_ratio < 0.2:
        label = 0  # neither
    elif ends_ies or (ends_s and not word.endswith("ss")) or ends_es:
        label = 2  # plural
    else:
        label = 1  # singular
    
    d_features[word] = {
        "Length": length,
        "ends_s": ends_s,
        "ends_es": ends_es,
        "ends_ies": ends_ies,
        "Count": count,
        "is_alpha": int(is_alpha),
        "has_digit": has_digit,
        "has_nonalpha": has_nonalpha,
        "vowel_ratio": vowel_ratio,
        "max_consonant": max_consonant,
        "too_short_or_long": too_short_or_long,
        "Label": label
    }

df = pd.DataFrame(d_features).T
print(df)
x = df.drop(columns=["Label"]).values
y = df["Label"].values


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(x_train, y_train)

# Predictions
y_pred = regressor.predict(x_test)
y_pred_labels = np.rint(y_pred).astype(int)
y_pred_labels = np.clip(y_pred_labels, 0, 2)

# Metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
print("Accuracy:", accuracy_score(y_test, y_pred_labels))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_labels))
print("\nClassification Report:\n", classification_report(y_test, y_pred_labels))

# Interactive prediction
while True:
    word = input("Enter a word (or type 'no' to exit): ").strip()
    if word.lower() == 'no':
        print("Exiting...")
        break

    length = len(word)
    ends_s = int(word.endswith("s"))
    ends_es = int(word.endswith("es"))
    ends_ies = int(word.endswith("ies"))
    is_alpha = int(word.isalpha())
    has_digit = int(any(c.isdigit() for c in word))
    has_nonalpha = int(not word.isalpha())
    vowel_ratio = sum(c in "aeiou" for c in word.lower()) / max(1, length)
    max_consonant = max_consonant_cluster(word)
    too_short_or_long = int(length < 2 or length > 20)

    x_new = [[
        length,
        ends_s,
        ends_es,
        ends_ies,
        1,  # Count for unseen word
        is_alpha,
        has_digit,
        has_nonalpha,
        vowel_ratio,
        max_consonant,
        too_short_or_long
    ]]

    y_new_pred = regressor.predict(x_new)[0]
    y_new_label = int(np.clip(np.rint(y_new_pred), 0, 2))

    label_map = {0: "neither", 1: "singular", 2: "plural"}
    print(f"The word '{word}' is predicted as: {label_map[y_new_label]} (raw={y_new_pred:.2f})")