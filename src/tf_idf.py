import praw

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

from math import log
import numpy as np
from src.utils import get_dataset
from collections import Counter


######################
# PREPROCESSING
######################
def convert_lower_case(data):
    return np.char.lower(data)


def remove_unnecessary_symbols(data):
    symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
    for i in range(len(symbols)):
        data = np.char.replace(data, symbols[i], ' ')
        data = np.char.replace(data, "  ", " ")
    data = np.char.replace(data, ',', '')
    return data


def stemming(data):
    stemmer = PorterStemmer()

    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        new_text = new_text + " " + stemmer.stem(w)
    return new_text


def preprocess(data):
    data = convert_lower_case(data)
    data = stemming(data)
    data = remove_unnecessary_symbols(data)
    data = str(data).split()
    return data


######################
# INVERTED LIST
######################
def get_inverted_list(dataset):
    # Step 1: Tokenize the documents
    # Convert each document to lowercase and split it into words
    duplicate_tokens = []
    temp_duplicate_tokens = []
    for document in dataset:
        title_token = preprocess(document['title'])
        content_token = preprocess(document['content'])

        duplicate_tokens.extend(title_token)
        duplicate_tokens.extend(content_token)

        temp_duplicate_tokens.append(title_token + content_token)

    # Combine the tokens into a list of unique terms
    terms = list(set(duplicate_tokens))

    # Step 2: Build the inverted index
    # Create an empty dictionary to store the inverted index
    inverted_index = {}

    # For each term, find the documents that contain it
    for term in terms:
        documents = []

        for document in enumerate(temp_duplicate_tokens):
            if term in document[1]:
                documents.append(document[0])

        inverted_index[term] = documents

    # # Step 3: Print the inverted index
    # for term, documents in inverted_index.items():
    #     print(term, "->", documents)

    return inverted_index


######################
# TF-IDF
######################

# Function to calculate TF-IDF
def calculate_tf_idf(term, doc_id, document_dict):
    tf = document_dict[term].count(doc_id) / len(document_dict[term])
    idf = log(len(document_dict) / (len([1 for doc_ids in document_dict.values() if doc_id in doc_ids]) + 1))
    return tf * idf


def compute_query_tf_idf(query, N, df):
    """
    :param query: query string
    :param N: the total number of documents in your dataset.
    :param df: a dictionary where each key is a word and its value is the number of documents that contain that word.
    :return: tf_idf
    """
    tf_query = Counter(query)  # query tf
    tf_idf_query = {}
    for word, tf in tf_query.items():
        idf = log(N / df.get(word, N))  # Adding N in case the word is not in df
        tf_idf_query[word] = tf * idf
    return tf_idf_query


######################
# COSINE SIMILARITY
######################
def cosine_similarity(query_tfidf, document_tfidf):
    similarities = {}
    for doc_id, vec in document_tfidf.items():
        dot_product = np.dot(query_tfidf, vec)
        query_norm = np.linalg.norm(query_tfidf)
        document_norm = np.linalg.norm(vec)

        # Avoid division by zero
        denominator = x if (x := query_norm * document_norm) != 0 else 1

        cos_sim = dot_product / denominator
        similarities[doc_id] = cos_sim
    return similarities


# Function to rank documents based on a query
def rank_documents(query, document_dict, N):

    query_score = compute_query_tf_idf(query, N, {key: len(value) for key, value in document_dict.items()})

    # Calculate TF-IDF for each term in the query for each document
    document_scores = {}
    for i, term in enumerate(query):
        for doc_id in range(N):  # Assuming documents are numbered from 1 to 9
            if doc_id not in document_scores:
                document_scores[doc_id] = np.zeros(len(query))
            if term in document_dict:
                document_scores[doc_id][i] = calculate_tf_idf(term, doc_id, document_dict)

    document_scores = cosine_similarity(np.array(list(query_score.values())), document_scores)

    return sorted(document_scores.items(), key=lambda x: x[1], reverse=True)


def ranked_retrieval(dataset, query, threshhold):
    query = preprocess(query)
    words_set = get_inverted_list(dataset)
    # print(words_set)

    # Rank documents based on the query
    result = rank_documents(query, words_set, len(dataset))

    # Print the ranked documents
    # print(f"Ranked Documents for Query '{query}':")
    # for doc_id, score in filter(lambda x: x[1] > threshhold, result):
    #     print(f"Document {doc_id}: {score}")

    documents = []
    for doc_id, score in result:
        if score >= threshhold:
            documents.append(dataset[doc_id])
        break

    return documents


if __name__ == "__main__":
    SUBREDDIT = 'AITAH'
    LIMIT = 50

    query = "for telling my girlfriend she needs to enforce better boundaries with her daughter regarding entering our room"
    k = 10

    dataset = get_dataset(SUBREDDIT, LIMIT)
    ranked_retrieval(dataset, query, 0.5)
