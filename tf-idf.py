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


# TODO: set this in remove_unnecessary_symbols
def remove_apostrophe(data):
    return np.char.replace(data, "'", "")


def remove_stop_words(data):
    stop_words = stopwords.words('english')
    words = word_tokenize(str(data))
    new_text = ""
    for w in words:
        if w not in stop_words and len(w) > 1:
            new_text = new_text + " " + w
    return new_text


def stemming(data):
    stemmer = PorterStemmer()

    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        new_text = new_text + " " + stemmer.stem(w)
    return new_text


def convert_numbers(data):
    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        try:
            w = num2words(int(w))
        except:
            pass
        new_text = new_text + " " + w
    new_text = np.char.replace(new_text, "-", " ")
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
        #title_token = document['title'].lower() #.split()
        #content_token = document['content'].lower() #.split()

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


######################
# OTHER STEPS
######################
dataset = get_dataset()
temp = get_inverted_list(dataset)







#####################




N = len(dataset)

processed_text = []
processed_title = []
for i in dataset:
    processed_title.append(word_tokenize(str(preprocess(i['title']))))
    processed_text.append(word_tokenize(str(preprocess(i['content']))))

DF = {}
for i in range(N):
    text_tokens = processed_text[i]
    title_tokens = processed_title[i]
    for w in text_tokens:
        try:
            DF[w].add(i)
        except:
            DF[w] = {i}

    for w in title_tokens:
        try:
            DF[w].add(i)
        except:
            DF[w] = {i}
for i in DF:
    DF[i] = len(DF[i])

total_vocab_size = len(DF)

total_vocab = [x for x in DF]


def doc_freq(word):
    c = 0
    try:
        c = DF[word]
    except:
        pass
    return c


doc = 0

tf_idf = {}

for i in range(N):
    tokens = processed_text[i]

    counter = Counter(tokens + processed_title[i])
    words_count = len(tokens + processed_title[i])

    for token in np.unique(tokens):
        tf = counter[token] / words_count
        df = doc_freq(token)
        if df == 0:
            idf = 0
        else:
            idf = math.log(N / df)
        tf_idf[doc, token] = tf * idf
    doc += 1

doc = 0

tf_idf_title = {}

for i in range(N):
    tokens = processed_title[i]
    counter = Counter(tokens + processed_text[i])
    words_count = len(tokens + processed_text[i])

    for token in np.unique(tokens):
        tf = counter[token] / words_count
        df = doc_freq(token)
        if df == 0:
            idf = 0
        else:
            idf = math.log(N / df)
        tf_idf_title[doc, token] = tf * idf
    doc += 1

for i in tf_idf:
    tf_idf[i] *= alpha

for i in tf_idf_title:
    tf_idf[i] = tf_idf_title[i]

D = np.zeros((N, total_vocab_size))
for i in tf_idf:
    try:
        ind = total_vocab.index(i[1])
        D[i[0]][ind] = tf_idf[i]
    except:
        pass


######################
# COSINE SIMILARITY
######################
def gen_vector(tokens):
    Q = np.zeros((len(total_vocab)))

    counter = Counter(tokens)
    words_count = len(tokens)

    for token in np.unique(tokens):
        tf = counter[token] / words_count
        df = doc_freq(token)
        if df == 0:
            idf = 0
        else:
            idf = math.log(N / df)

        try:
            ind = total_vocab.index(token)
            Q[ind] = tf * idf
        except:
            pass
    return Q


def cosine_similarity():
    preprocessed_query = preprocess(query)
    tokens = word_tokenize(str(preprocessed_query))

    print("Tokens:", tokens)

    d_cosines = []

    query_vector = gen_vector(tokens)

    for d in D:
        cosine = np.dot(query_vector, d) / (np.linalg.norm(query_vector) * np.linalg.norm(d))
        d_cosines.append(cosine)
    return d_cosines


def get_indexes(list):
    list_tmp = list.copy()
    list_sorted = list.copy()
    list_sorted.sort()

    list_index = []
    for x in list_sorted:
        list_index.insert(0, list_tmp.index(x))
        list_tmp[list_tmp.index(x)] = -1

    return list_index


def ranked_retrieval(Q, dataset, k):
    print("Query:", query)

    print("-" * 50)
    print("Cosine Similarity")
    print("-" * 50)

    P = get_indexes(Q)

    print("\nCosine similarities:", Q)

    print("\nindexes:", P)

    for i in range(k):
        print("")
        print(f"Rank {i}:")
        print(f"    title: {dataset[P[i]]}")
        print(f"    cosine similarity: {Q[P[i]]}")


# Main Execution
#ranked_retrieval()

cosine_similarities = cosine_similarity()
ranked_retrieval(cosine_similarities, dataset, k)