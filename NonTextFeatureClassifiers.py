import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix 
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_parquet("smalldata.parquet")
# Define input features and target
X = df[[
    'text_length', 'word_count', 'avg_word_length',
    'punctuation_count', 'uppercase_count',
    'special_char_count', 'sentence_count', 'words_per_sentence'
]]
y = df['label']

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "Support Vector Machine": SVC()
}   

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n{name} Results:")
    print(classification_report(y_test, y_pred, target_names=['Human', 'AI']))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


print("Done")
