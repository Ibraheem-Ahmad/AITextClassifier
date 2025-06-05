import pandas as pd
import numpy as np
from scipy.sparse import hstack
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler




# ------------------------
# Load Dataset
# ------------------------
df = pd.read_parquet("smalldata.parquet")


engineered_features = [
    'text_length', 'word_count', 'avg_word_length',
    'punctuation_count', 'uppercase_count',
    'special_char_count', 'sentence_count', 'words_per_sentence'
]


X_engineered = df[engineered_features]
y = df['label']
X_text = df['text']

# Split both text and style features
X_text_train, X_text_test, X_eng_train, X_eng_test, y_train, y_test = train_test_split(
    X_text, X_engineered, y, test_size=0.2, stratify=y, random_state=42
)

# ------------------------
# TF-IDF Vectorization
# ------------------------
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    ngram_range=(1, 2)
)
X_text_train_vec = vectorizer.fit_transform(X_text_train)
X_text_test_vec = vectorizer.transform(X_text_test)

# ------------------------
# Combine Text + Engineered Features
# ------------------------
scaler = StandardScaler()
X_eng_train_scaled = scaler.fit_transform(X_eng_train)
X_eng_test_scaled = scaler.transform(X_eng_test)

# Then combine with TF-IDF
X_train_combined = hstack([X_text_train_vec, X_eng_train_scaled])
X_test_combined = hstack([X_text_test_vec, X_eng_test_scaled])

#X_train_combined = hstack([X_text_train_vec, np.array(X_eng_train)])
#X_test_combined = hstack([X_text_test_vec, np.array(X_eng_test)])

# ------------------------
# Train and Evaluate Models
# ------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, n_jobs=-1),
    "Support Vector Machine": LinearSVC(),
    "Naive Bayes": MultinomialNB()
}

for name, model in models.items():
    print(f"\n==== {name} ====")
    model.fit(X_train_combined, y_train)
    y_pred = model.predict(X_test_combined)

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Human', 'AI']))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Logistic Regression top words
    if name == "Logistic Regression":
        feature_names = vectorizer.get_feature_names_out()
        coefs = model.coef_[0][:len(feature_names)]  # only the TF-IDF part

        top_ai = sorted(zip(coefs, feature_names), reverse=True)[:10]
        top_human = sorted(zip(coefs, feature_names))[:10]

        print("\nTop Indicative Words for AI:")
        for coef, word in top_ai:
            print(f"{word}: {coef:.3f}")

        print("\nTop Indicative Words for Human:")
        for coef, word in top_human:
            print(f"{word}: {coef:.3f}")
