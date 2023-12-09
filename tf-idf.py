import praw

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

from math import log
import numpy as np

######################
# PARAMETERS
######################
CLIENT_ID = 'Ho0YiNH572NQcYCMcM3rGQ'
CLIENT_SECRET = 'jLc5u4v3jH7O8ijnICUyMaPBDnG7qA'
USER_AGENT = 'mozilla:com.example.sentiment-analysis:v1 (by u/def-not-bot-420)'
USERNAME = 'def-not-bot-420'
PASSWORD = 'IrIsCool69'

SUBREDDIT = 'AITAH'
LIMIT = 50

alpha = 0.3
query = "I want to go in the doorway"
k = 10


######################
# DATASET
######################
def get_dataset():
    # Set up PRAW with your credentials
    reddit = praw.Reddit(client_id=CLIENT_ID,
                         client_secret=CLIENT_SECRET,
                         user_agent=USER_AGENT,
                         username=USERNAME,
                         password=PASSWORD)

    # Choose the subreddit
    subreddit = reddit.subreddit(SUBREDDIT)  # Replace 'subreddit_name' with your target subreddit

    # Fetch the top 10 hot posts
    dataset = []
    for post in subreddit.hot(limit=LIMIT):
        dataset.append({'title': post.title, 'content': post.selftext})
    return dataset


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

    # Step 3: Print the inverted index
    for term, documents in inverted_index.items():
        print(term, "->", documents)

    return inverted_index


######################
# PUTTING IT ALL TOGETHER
######################

# Function to calculate TF-IDF
def calculate_tf_idf(term, doc_id, document_dict):
    tf = document_dict[term].count(doc_id) / len(document_dict[term])
    idf = log(len(document_dict) / len([1 for doc_ids in document_dict.values() if doc_id in doc_ids]))
    tf_idf = tf * idf
    return tf_idf


def cosine_similarity(query_tfidf, document_tfidf):
    dot_product = np.dot(query_tfidf, document_tfidf)
    query_norm = np.linalg.norm(query_tfidf)
    document_norm = np.linalg.norm(document_tfidf)

    # Avoid division by zero
    denominator = query_norm * document_norm if query_norm * document_norm != 0 else 1

    cos_sim = dot_product / denominator
    return cos_sim


# Function to rank documents based on a query
def rank_documents(query, document_dict):
    query_terms = query.split()

    # Calculate TF-IDF for each term in the query for each document
    document_scores = {}
    for term in query_terms:
        for doc_id in range(0, LIMIT):  # Assuming documents are numbered from 1 to 9
            if doc_id not in document_scores:
                document_scores[doc_id] = 0
            if term in document_dict:
                document_scores[doc_id] += calculate_tf_idf(term, doc_id, document_dict)


    # Sort documents based on the total TF-IDF score
    ranked_documents = sorted(document_scores.items(), key=lambda x: x[1], reverse=True)

    return ranked_documents


def ranked_retrieval(dataset, query):
    words_set = get_inverted_list(dataset)
    print(words_set)

    # Rank documents based on the query
    result = rank_documents(query, words_set)

    # Print the ranked documents
    print(f"Ranked Documents for Query '{query}':")
    for doc_id, score in result:
        print(f"Document {doc_id}: {score}")


dataset = get_dataset()
ranked_retrieval(dataset, query)
