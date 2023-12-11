import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, recall_score
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import random
from tfidf import ranked_retrieval

# Load the dataset
newsgroups = fetch_20newsgroups(subset='all')
data = newsgroups.data
target = newsgroups.target


# Preprocessing functions
def preprocess(text):
    stemmer = PorterStemmer()
    text = text.lower()
    tokens = word_tokenize(text)
    stemmed = [stemmer.stem(token) for token in tokens]
    return ' '.join(stemmed)


# Preprocess the data
processed_data = [preprocess(text) for text in data]

print(processed_data)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(processed_data)


# Function to simulate a query (for demonstration)
def simulate_query(category):
    return newsgroups.data[random.choice(np.where(target == category)[0])]


# Pick a category as the 'query'
category = 10  # change as needed
query = simulate_query(category)
query = preprocess(query)

# Vectorize the query
query_vec = vectorizer.transform([query])

# Compute cosine similarity between query and all documents
cosine_similarities = np.dot(query_vec, vectors.T).toarray()[0]

# Select top 5 documents
top5_idx = np.argsort(cosine_similarities)[-5:]

# Evaluate Precision and Recall
true_labels = (target == category).astype(int)
predicted_labels = np.zeros_like(true_labels)
predicted_labels[top5_idx] = 1

precision = precision_score(true_labels, predicted_labels)
recall = recall_score(true_labels, predicted_labels)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
