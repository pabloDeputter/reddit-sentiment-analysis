import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')
nltk.download('punkt')


def preprocess_text(text):
    # Tokenize and remove punctuation
    tokens = nltk.word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalnum()]
    return ' '.join(tokens)


def bert_sentiment_analysis(text):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)  # Assuming 3 labels for sentiment analysis

    # Tokenize and prepare input data
    tokenized_text = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors='pt')
    input_ids = tokenized_text['input_ids']

    # Get sentiment prediction
    with torch.no_grad():
        outputs = model(**tokenized_text)
    predictions = torch.argmax(outputs.logits, dim=1).item()

    return predictions


def ranked_retrieval_with_sentiment(query, documents):
    # Preprocess query
    preprocessed_query = preprocess_text(query)

    # Preprocess and vectorize documents
    preprocessed_documents = [preprocess_text(doc['title'] + ' ' + doc['content']) for doc in documents]

    vectorizer = TfidfVectorizer()
    document_vectors = vectorizer.fit_transform(preprocessed_documents)

    # Vectorize the query
    query_vector = vectorizer.transform([preprocessed_query])

    # Calculate cosine similarity between query and documents
    similarities = cosine_similarity(query_vector, document_vectors).flatten()

    # Rank documents based on similarity
    ranked_indices = similarities.argsort()[::-1]

    # Display ranked results with sentiment analysis
    for i, index in enumerate(ranked_indices):
        doc = documents[index]
        title = doc['title']
        content = doc['content']

        # Perform sentiment analysis using BERT
        sentiment_prediction = bert_sentiment_analysis(title + ' ' + content)

        sentiment_labels = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

        print(f"Rank {i + 1}: {title} - Similarity: {similarities[index]:.4f} - Sentiment: {sentiment_labels[sentiment_prediction]}")
        print(f"  Content: {content}")
        print()


# Example database of documents
documents = [
    {'title': 'Document 1',
     'content': 'This is the content of Document 1. It contains information about a happy topic.'},
    {'title': 'Document 2',
     'content': 'Document 2 is another example with some text in its content, but it discusses a sad event.'},
    # Add more documents as needed
]

# Example query
query = 'happy topic'

# Perform ranked retrieval with sentiment analysis
ranked_retrieval_with_sentiment(query, documents)
