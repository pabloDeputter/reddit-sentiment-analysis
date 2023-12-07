import json

import dotenv
import numpy as np
from flask import Flask, render_template, request, jsonify
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer

from utils import get_dataset

dotenv.load_dotenv()

app = Flask(__name__)

# load tokenizer and model, create trainer
model_name = "SamLowe/roberta-base-go_emotions"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
trainer = Trainer(model=model)


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
    subreddit = request.args.get('subreddit', 'all')
    posts = get_dataset(subreddit, 10)

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

    for post, res in zip(posts, result):
        post['emotion'] = json.dumps([res])
    return jsonify({'posts': posts})


@app.route('/api/posts/<string:query>')
def get_new_posts(query):
    print(query)
    category = request.args.get('subreddit', 'all')

    posts = get_dataset(category, 10)

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

    for post, res in zip(posts, result):
        post['emotion'] = json.dumps([res])
    return jsonify({'posts': posts})



if __name__ == '__main__':
    app.run(debug=True, port=8000)
