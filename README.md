# Sentiment Analysis with Naive Bayes

This project involves sentiment analysis using the Naive Bayes classification algorithm to classify text data into three sentiment categories: Positive, Neutral, and Negative. The project encompasses data loading, preprocessing, model training, evaluation, and optional model saving.

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Results and Visualization](#results-and-visualization)
5. [Optional: Model Saving](#optional-model-saving)

## Introduction

Sentiment analysis is a natural language processing task that involves determining the sentiment expressed in a piece of text. This project utilizes the Naive Bayes classification algorithm to perform sentiment analysis on Twitter data.

## Installation

To run this project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/Vaishnavilily/Coding-Raja_Technologies-Internship.git
   cd your-repository
2. Install the required dependencies:
   ```bash
   pip install pandas numpy nltk scikit-learn matplotlib seaborn
3. Download NLTK data:
   ```bash
   python -m nltk.downloader stopwords wordnet
4. Run the Script:
   ```bash
   python sentiment_analysis.py

## Usage

### Loading And Preprocessing Data

Data Loading
The script reads sentiment data from the 'Twitter_data.csv' file using Pandas.
```python
import pandas as pd

# Load the dataset
df = pd.read_csv('Twitter_data.csv')
```
1. Text Preprocessing
2. Lowercasing
3. Removing URLs, emojis, and non-alphanumeric characters
4. Lemmatization
5. Removing stop words

```python
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = ' '.join(lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words)
        return text
    else:
        return str(text)  # Convert non-string types to string

df['text'] = df['clean_text'].apply(preprocess_text)
```

### Model Training and Evaluation

The dataset is split into training and testing sets using a 50-50 split ratio.
Text data is vectorized using the CountVectorizer, and a Naive Bayes model (ComplementNB) is trained on the vectorized training data.

```python
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
)
import matplotlib.pyplot as plt
import seaborn as sns

# Split into train and test sets
train_df, test_df = train_test_split(df, test_size=0.5, random_state=42)

# Vectorize the text data
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_df['text'])
X_test = vectorizer.transform(test_df['text'])

# Train Naive Bayes model
nb_model = ComplementNB()
nb_model.fit(X_train, train_df['sentiment'])
```

### Evaluation Metrics

The script calculates and prints various evaluation metrics, including accuracy, precision, recall, and F1 score. It also generates a classification report and a confusion matrix.

```python
# Make predictions on the test set
predictions = nb_model.predict(X_test)

# Calculate accuracy, precision, recall, and F1 score
accuracy = accuracy_score(test_df['sentiment'], predictions)
precision = precision_score(test_df['sentiment'], predictions, average='weighted')
recall = recall_score(test_df['sentiment'], predictions, average='weighted')
f1 = f1_score(test_df['sentiment'], predictions, average='weighted')

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')

# Classification Report and Confusion Matrix
print('Classification Report:\n', classification_report(test_df['sentiment'], predictions))
conf_matrix = confusion_matrix(test_df['sentiment'], predictions)
print('Confusion Matrix:\n', conf_matrix)
```

## Results and Visualization

### Confusion Matrix

A confusion matrix is plotted to visualize model performance on the test set.

```python
# Plot Confusion Matrix
fig, ax = plt.subplots(figsize=(8, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=nb_model.classes_, yticklabels=nb_model.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
```

### Precision-Recall Curves

Precision-Recall curves are plotted for each sentiment category (Positive, Neutral, Negative) to provide insights into the model's precision and recall across different thresholds.

```python
# Plot Precision-Recall Curve
precision, recall, _ = precision_recall_curve((test_df['sentiment'] == 'Positive').astype(int), nb_model.predict_proba(X_test)[:, 2])
plt.figure(figsize=(8, 8))
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Positive Sentiment Precision-Recall Curve')
plt.show()

# Plot Precision-Recall Curve
precision, recall, _ = precision_recall_curve((test_df['sentiment'] == 'Neutral').astype(int), nb_model.predict_proba(X_test)[:, 2])
plt.figure(figsize=(8, 8))
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Neutral Sentiment Precision-Recall Curve')
plt.show()

# Plot Precision-Recall Curve
precision, recall, _ = precision_recall_curve((test_df['sentiment'] == 'Negative').astype(int), nb_model.predict_proba(X_test)[:, 2])
plt.figure(figsize=(8, 8))
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Negative Sentiment Precision-Recall Curve ')
plt.show()
```

## Optional: Model Saving

The trained model and vectorizer can be saved using joblib for later use.

```python
# Save the trained model (optional)
import joblib
joblib.dump(nb_model, 'naive_bayes_model.joblib')
joblib.dump(vectorizer, 'vectorizer.joblib')
```
