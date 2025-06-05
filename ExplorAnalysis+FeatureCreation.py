import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the dataset
df = pd.read_parquet("smalldata.parquet")

# Print basic info
print("Dataset Info:")
df.info()

# Sample data
print("\nSample Data:")
print(df.sample(5))

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Source distribution (to understand label breakdown)
print("\nSource Value Counts:")
print(df['source'].value_counts())

# Anything not labeled "Human" is AI
df['label'] = df['source'].apply(lambda x: 0 if x == "Human" else 1)


sns.countplot(x='label', data=df)
plt.title('Label Distribution (0 = Human, 1 = AI)')
plt.xlabel('Label')
plt.ylabel('Count')
plt.xticks([0, 1], ['Human', 'AI'])
plt.show()

# Average word length
df['avg_word_length'] = df['text'].apply(lambda x: np.mean([len(word) for word in x.split()]) if len(x.split()) > 0 else 0)

# Punctuation count
df['punctuation_count'] = df['text'].apply(lambda x: sum([1 for c in x if c in '.,;:!?']))

# Uppercase letter count
df['uppercase_count'] = df['text'].apply(lambda x: sum(1 for c in x if c.isupper()))

# Special character count (non-alphanumeric)
df['special_char_count'] = df['text'].apply(lambda x: sum(1 for c in x if not c.isalnum() and not c.isspace()))

# Sentence count (rough estimate based on period)
df['sentence_count'] = df['text'].apply(lambda x: x.count('.'))

# Word density (words per sentence)
df['words_per_sentence'] = df['word_count'] / df['sentence_count'].replace(0, 1)

# Select features to inspect
features = ['text_length', 'word_count', 'avg_word_length', 'punctuation_count',
            'uppercase_count', 'special_char_count', 'sentence_count', 'words_per_sentence', 'label']

# Correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(df[features].corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()

df.to_parquet("smalldata.parquet", index=False)

