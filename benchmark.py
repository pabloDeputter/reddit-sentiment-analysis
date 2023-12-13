import re

from sklearn.datasets import fetch_20newsgroups
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score

import tf_idf

# Load the dataset
newsgroups = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))

# Extract the documents and their corresponding labels
documents = newsgroups.data
labels = newsgroups.target

# Preprocess all documents
preprocessed_documents = [' '.join(tf_idf.preprocess(doc)) for doc in documents]
queries = ["computer hardware", "middle east conflict", "space exploration"]


print("TF-IDF")
ranked_results = {}
for query in queries:
    ranked_documents = tf_idf.ranked_retrieval(preprocessed_documents, query, 0.5)
    ranked_results[query] = ranked_documents

vectorizer = TfidfVectorizer()
X_ref = vectorizer.fit_transform(preprocessed_documents)
print("done")



def evaluate(your_ranked_documents, ref_ranked_documents, labels):
    # Implement the logic to calculate Precision, Recall, F1-Score, nDCG
    # You might need to modify this depending on how your results are structured
    precision = precision_score(ref_ranked_documents, your_ranked_documents)
    recall = recall_score(ref_ranked_documents, your_ranked_documents)
    f1 = f1_score(ref_ranked_documents, your_ranked_documents)

    return precision, recall, f1



print("Sklearn TF-IDF")
ref_ranked_results = {}
for query in queries:
    query_vector = vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_vector, X_ref).flatten()
    ranked_docs = np.argsort(cosine_similarities, axis=0)[::-1]
    ref_ranked_results[query] = ranked_docs

# Call the evaluate function
precision, recall, f1 = evaluate(ranked_results, ref_ranked_results, labels)
print(f'Precision: {precision}, Recall: {recall}, F1-Score: {f1}')
