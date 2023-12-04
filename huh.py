import praw

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import Counter
from num2words import num2words

import math

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
LIMIT = 10

alpha = 0.3
query = "parents over"
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


def get_words(doc):
    return



######################
# PUTTING IT ALL TOGETHER
######################
import pandas as pd


dataset = get_dataset()
words_set = get_inverted_list(dataset)
print(words_set)

n_docs = len(dataset)  # ·Number of documents in the corpus
n_words_set = len(words_set)  # ·Number of unique words in the

df_tf = pd.DataFrame(np.zeros((n_docs, n_words_set)), columns=words_set.keys())

print(df_tf)

# Compute Term Frequency (TF)
for i in range(n_docs):
    words = get_words(i)

    words = dataset[i].split(' ')  # Words in the document
    for w in words:
        df_tf[w][i] = df_tf[w][i] + (1 / len(words))

df_tf