import pandas as pd
import numpy as np
# import inflect
# import enchant

# p = inflect.engine()
# d_enchant = enchant.Dict("en_US")

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

# Output: dictionary of dictionaries
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

    # if length <= 2 or not is_alpha:
    #     label = 0   # neither
    # elif ends_ies or (ends_s and not word.endswith("ss")) or ends_es:
    #     label = 2   # plural
    # else:
    #     label = 1   # singular

    if has_digit or has_nonalpha or too_short_or_long or max_consonant > 10 or vowel_ratio < 0.2:
        label = 0  # neither
    elif ends_ies or (ends_s and not word.endswith("ss")) or ends_es:
        label = 2  # plural
    else:
        label = 1  # singular
    
    # Labeling using enchant + inflect + vowel_ratio + max_consonant
    # if (not d_enchant.check(word)  # not in dictionary
    #     or has_digit
    #     or has_nonalpha
    #     or too_short_or_long
    #     or vowel_ratio < 0.2       # low vowel ratio → likely gibberish
    #     or max_consonant > 7):     # very long consonant cluster → likely gibberish
    #     label = 0  # neither
    # elif p.singular_noun(word):
    #     label = 2  # plural
    # else:
    #     label = 1  # singular

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

# Print results
# for word, features in d_features.items():
#     print(word, "->", features)

df = pd.DataFrame(d_features).T
print(df)
x = df.drop(columns=["Label"]).values
y = df["Label"].values
# print(x[:20])

# Train-test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Check first 20 rows of x_train
# print("x_train sample:\n", x_train[:80])

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0, max_iter=10000)
classifier.fit(x_train, y_train)


y_pred = classifier.predict(x_test)
# print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

from sklearn.metrics import confusion_matrix, accuracy_score
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

while True:
    word = input("Enter a word (or type 'no' to exit): ").strip()
    if word.lower() == 'no':
        print("Exiting...")
        break

    length = len(word)
    is_alpha = int(all(c.isalpha() and c.isascii() for c in word))
    has_digit = int(any(c.isdigit() for c in word))
    has_nonalpha = int(not is_alpha)
    vowel_ratio = sum(c in "aeiou" for c in word.lower()) / max(1,length)
    max_consonant = max_consonant_cluster(word)
    too_short_or_long = int(length < 2 or length > 20)

    x_new = [[
        length,
        int(word.endswith("s")),
        int(word.endswith("es")),
        int(word.endswith("ies")),
        is_alpha,
        1,  # count
        has_digit,
        has_nonalpha,
        vowel_ratio,
        max_consonant,
        too_short_or_long
    ]]
    pred = classifier.predict(x_new)[0]
    label_map = {0: "neither", 1: "singular", 2: "plural"}
    print(f"The word '{word}' is predicted as: {label_map[pred]}")