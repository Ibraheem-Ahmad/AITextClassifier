import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

# ------------------------
# Load Data
# ------------------------
df = pd.read_parquet("smalldata.parquet")

# Input/output
X_text = df['text']
y = df['label']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_text, y, test_size=0.2, stratify=y, random_state=42
)

# ------------------------
# TF-IDF Vectorization
# ------------------------
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    ngram_range=(1, 2)
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ------------------------
# Define and Train Models
# ------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, n_jobs=-1),
    "Support Vector Machine": LinearSVC(),
    "Naive Bayes": MultinomialNB()
}

for name, model in models.items():
    print(f"\n==== {name} ====")
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Human', 'AI']))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Show top words for Logistic Regression only (has coef_)
    if name == "Logistic Regression":
        feature_names = vectorizer.get_feature_names_out()
        coefs = model.coef_[0]
        top_ai = sorted(zip(coefs, feature_names), reverse=True)[:10]
        top_human = sorted(zip(coefs, feature_names))[:10]

        print("\nTop Indicative Words for AI:")
        for coef, word in top_ai:
            print(f"{word}: {coef:.3f}")

        print("\nTop Indicative Words for Human:")
        for coef, word in top_human:
            print(f"{word}: {coef:.3f}")
