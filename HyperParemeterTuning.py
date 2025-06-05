import pandas as pd
import numpy as np
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load preprocessed dataset with engineered features and label
df = pd.read_parquet("smalldata.parquet")

# Define text and engineered features
engineered_cols = [
    'text_length', 'word_count', 'avg_word_length',
    'punctuation_count', 'uppercase_count',
    'special_char_count', 'sentence_count', 'words_per_sentence'
]
X_text = df['text']
X_engineered = df[engineered_cols]
y = df['label']

# Train-test split
X_text_train, X_text_test, X_eng_train, X_eng_test, y_train, y_test = train_test_split(
    X_text, X_engineered, y, test_size=0.2, stratify=y, random_state=42
)

# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
X_text_train_vec = vectorizer.fit_transform(X_text_train)
X_text_test_vec = vectorizer.transform(X_text_test)

# Scale engineered features
scaler = StandardScaler()
X_eng_train_scaled = scaler.fit_transform(X_eng_train)
X_eng_test_scaled = scaler.transform(X_eng_test)

# Combine features
X_train_combined = hstack([X_text_train_vec, X_eng_train_scaled])
X_test_combined = hstack([X_text_test_vec, X_eng_test_scaled])

# ---------------------
# Logistic Regression Tuning
# ---------------------
logistic_params = {
    'C': [0.1, 1, 10],
    'solver': ['lbfgs', 'liblinear'],
    'max_iter': [1000, 2000]
}
logistic_grid = GridSearchCV(
    LogisticRegression(),
    logistic_params,
    scoring='f1_weighted',
    cv=3,
    verbose=1
)
logistic_grid.fit(X_train_combined, y_train)
print("Best Logistic Regression Params:", logistic_grid.best_params_)
y_pred_log = logistic_grid.predict(X_test_combined)
print("Logistic Regression Classification Report:")
print(classification_report(y_test, y_pred_log, target_names=['Human', 'AI']))

# ---------------------
# Random Forest Tuning
# ---------------------
rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [None, 20, 40],
    'min_samples_split': [2, 5],
    'max_features': ['sqrt', 'log2']
}
rf_grid = GridSearchCV(
    RandomForestClassifier(n_jobs=-1, random_state=42),
    rf_params,
    scoring='f1_weighted',
    cv=3,
    verbose=1
)
rf_grid.fit(X_train_combined, y_train)
print("Best Random Forest Params:", rf_grid.best_params_)
y_pred_rf = rf_grid.predict(X_test_combined)
print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf, target_names=['Human', 'AI']))
