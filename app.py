import json
import dotenv
import numpy as np
from flask import Flask, render_template, request, jsonify
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer

import src.utils as utils
from tf_idf import ranked_retrieval

dotenv.load_dotenv()

app = Flask(__name__)

# load tokenizer and model, create trainer
model_name = "SamLowe/roberta-base-go_emotions"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
trainer = Trainer(model=model)

# Cache
cache_suggestions_FILE = 'data/cache_suggestions.pkl'
cache_suggestions = utils.load_cache_from_file(cache_suggestions_FILE)


# Create class for data preparation
class Posts(Dataset):
    def __init__(self, tokenized_texts):
        self.tokenized_texts = tokenized_texts

    def __len__(self):
        return len(self.tokenized_texts["input_ids"])

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.tokenized_texts.items()}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/posts')
def get_posts():
    """
    Returns a list of posts from the specified subreddit.
    subreddit: subreddit name
    emotion: emotion name
    num_posts: number of posts to return
    query: query parameter
    """
    subreddit = request.args.get('subreddit', 'all')
    emotion = request.args.get('emotion', 'all')
    num_posts = request.args.get('num_posts', 10)
    query = request.args.get('query', '')
    threshold = request.args.get('threshold', 0.5)

    # Retrieve posts from specified subreddit
    posts = utils.get_dataset(subreddit, int(num_posts))

    # Run TF-IDF
    if query != '':
        try:
            ranked_posts = list(ranked_retrieval(posts, query, float(threshold)))
        except ZeroDivisionError:
            return jsonify({'error': 'ZeroDivisionError, please try again with other parameters.'}), 400

        print(posts)
        print(ranked_posts)
        # Assign scores to posts, defaulting to 0
        for index, score in ranked_posts:
            posts[index]['score'] = score
        for i in range(len(posts)):
            if i not in [idx for idx, _ in ranked_posts]:
                posts[index]['score'] = 0

        return jsonify({'posts': json.dumps(posts)})

    pred_texts = [post['content'] for post in posts]
    # Tokenize texts and create prediction data set
    tokenized_texts = tokenizer(pred_texts, truncation=True, padding=True)
    pred_dataset = Posts(tokenized_texts)

    # Run predictions
    predictions = trainer.predict(pred_dataset)

    # scores raw
    temp = (np.exp(predictions[0]) / np.exp(predictions[0]).sum(-1, keepdims=True))

    result = [[{'label': label, 'score': float(score)} for label, score in zip(model.config.id2label.values(), scores)]
              for scores in temp]

    # Sort by emotion score
    if emotion != 'all':
        def get_emotion_score(post_emotions, label):
            for item in post_emotions:
                if item['label'] == label:
                    return item['score']
            return -1  # Default score if the emotion isn't found

        zipped = sorted(
            zip(posts, result),
            key=lambda x: get_emotion_score(x[1], emotion),
            reverse=True
        )
    else:
        zipped = zip(posts, result)

    posts, emotions = zip(*zipped)

    for post, emotion_scores in zip(posts, emotions):
        post['emotion'] = json.dumps(emotion_scores)

    return jsonify({'posts': list(posts)})


@app.route('/api/subreddits', methods=['GET'])
def autocomplete():
    """
    Returns a list of subreddit suggestions based on the query parameter
    '/api/subreddits?query=ask' -> ['askreddit', 'askscience', 'askhistorians', ...]
    """
    query = request.args.get('query', '')
    if query in cache_suggestions:
        return jsonify(cache_suggestions[query])
    else:
        suggestions = utils.get_subreddit_suggestions(query)
        cache_suggestions[query] = suggestions
        utils.save_cache_to_file(cache_suggestions, cache_suggestions_FILE)
        return jsonify(suggestions)


if __name__ == '__main__':
    app.run(debug=True, port=8000)
