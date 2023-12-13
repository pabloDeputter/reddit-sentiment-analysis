from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import ndcg_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score

import numpy as np
from tqdm import tqdm

import tf_idf


def calculate_ndcg(your_ranks, ref_ranks, k=10):
    """
    Calculate the Normalized Discounted Cumulative Gain (nDCG) at rank k.
    :param your_ranks: Your list of ranked document indices.
    :param ref_ranks: The reference list of ranked document indices (ideal ranking).
    :param k: The number of top elements to consider in the ranking.
    :return: The nDCG value.
    """

    def dcg_at_k(ranks, k):
        """Calculate DCG at rank k."""
        ranks = np.asfarray(ranks)[:k]
        return np.sum(ranks / np.log2(np.arange(2, ranks.size + 2)))

    # Create binary relevancy scores for your ranks and reference ranks
    rel_scores = np.isin(ref_ranks, your_ranks[:k]).astype(int)
    ideal_rel_scores = np.ones_like(rel_scores)

    # Calculate DCG and IDCG
    dcg = dcg_at_k(rel_scores, k)
    idcg = dcg_at_k(ideal_rel_scores, k)

    # Handle the case when IDCG is zero
    return dcg / idcg if idcg > 0 else 0


def tf_idf_implementation(dataset, queries):
    N = len(dataset)
    # Step 1: Tokenize the documents
    # Convert each document to lowercase and split it into words
    duplicate_tokens = []
    temp_duplicate_tokens = []
    for document in tqdm(dataset, desc="Tokenizing documents"):
        content_token = tf_idf.preprocess(document)
        duplicate_tokens.extend(content_token)
        temp_duplicate_tokens.append(content_token)

    # Combine the tokens into a list of unique terms
    terms = list(set(duplicate_tokens))

    # Step 2: Build the inverted index
    # Create an empty dictionary to store the inverted index
    inverted_index = {}

    # For each term, find the documents that contain it
    for term in tqdm(terms, desc="Building inverted index"):
        documents = [document[0] for document in enumerate(temp_duplicate_tokens) if term in document[1]]
        inverted_index[term] = documents

    ranked_results = {}
    for query in tqdm(queries, desc="Ranking documents"):
        preprocessed_query = tf_idf.preprocess(query)
        # Rank documents based on the query
        ranked_documents = tf_idf.rank_documents(preprocessed_query, inverted_index, N)
        ranked_results[query] = ranked_documents

    return ranked_results


def ref_preprocess(data):
    data = tf_idf.convert_lower_case(data)
    data = tf_idf.remove_unnecessary_symbols(data)
    data = tf_idf.stemming(data)
    # join into a single string
    return ' '.join(str(data).split())


# Load the dataset
newsgroups = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))

# Extract the documents and their corresponding labels
documents = newsgroups.data[:100]
labels = newsgroups.target

queries = ["computer hardware", "middle east conflict", "space exploration", "graphics", "university education",
           "atheists", "god", "nasa", "orbit", "moon", "sun", "thanks", "religion", "christian", "bible", "jesus",
           "windows", "software", "government", "interested", "medical advandes", "sports championships",
           "autmobile engineering"]

# Run TF-IDF own implementation
print("TF-IDF - own implementation")
ranked_results_own = tf_idf_implementation(documents, queries)

# Run TF-IDF sklearn implementation
print("TF-IDF - sklearn implementation")
preprocessed_documents = [ref_preprocess(doc) for doc in tqdm(documents, desc="Preprocessing documents")]
vectorizer = TfidfVectorizer()
X_ref = vectorizer.fit_transform(preprocessed_documents)
ranked_results_ref = {}
for query in tqdm(queries, desc="Ranking documents"):
    query_vector = vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_vector, X_ref).flatten()
    ranked_docs = np.argsort(cosine_similarities)[::-1]
    ranked_results_ref[query] = ranked_docs


def evaluate(your_ranked_documents, ref_ranked_documents, labels):
    # Implement the logic to calculate Precision, Recall, F1-Score, nDCG
    # You might need to modify this depending on how your results are structured
    precision = precision_score(ref_ranked_documents, your_ranked_documents)
    recall = recall_score(ref_ranked_documents, your_ranked_documents)
    f1 = f1_score(ref_ranked_documents, your_ranked_documents)

    return precision, recall, f1


# Call the evaluate function
# precision, recall, f1 = evaluate(ranked_results, ref_ranked_results, labels)
# print(f'Precision: {precision}, Recall: {recall}, F1-Score: {f1}')

def evaluate(your_ranked_documents, ref_ranked_documents):
    # Convert your ranked documents from [(index, score), ...] to [index, ...]
    your_ranked_indices = [doc[0] for doc in your_ranked_documents]

    return calculate_ndcg(your_ranked_indices, ref_ranked_documents)


average_ndcg = np.mean([evaluate(ranked_results_own[query], ranked_results_ref[query]) for query in queries])
print(f'Average nDCG: {average_ndcg}')
# Average nDCG: 0.01899946213775151
