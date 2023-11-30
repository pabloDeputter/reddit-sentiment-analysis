import json

from flask import Flask, render_template, request, jsonify
import praw
import os
import dotenv
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer


dotenv.load_dotenv()


app = Flask(__name__)

reddit = praw.Reddit(client_id=os.getenv('CLIENT_ID'), client_secret=os.getenv('CLIENT_SECRET'),
                     user_agent=os.getenv('USER_AGENT'), username=os.getenv('USERNAME'), password=os.getenv('PASSWORD'))

# load tokenizer and model, create trainer
model_name = "j-hartmann/emotion-english-distilroberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
trainer = Trainer(model=model)

# Create class for data preparation
class Posts:
    def __init__(self, tokenized_texts):
        self.tokenized_texts = tokenized_texts

    def __len__(self):
        return len(self.tokenized_texts["input_ids"])

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.tokenized_texts.items()}

@app.route('/')
def index():
    return render_template('index.html', post={'title': 'Geert Wilders wordt premier!!', 'selftext': 'This is the first post.',
                                               'comments': ["lol", "lollers"], 'sentiment': 4,
                                               'political_orientation': "hard right"})

# @app.route('/post/<post_id>')
# def post():
#     pass

@app.route('/back_posts')
def get_posts():
    category = request.args.get('category', 'all')

    posts = [
        {'title': 'First Post', 'content': 'This is the first post.'},
        {'title': 'Second Post', 'content': 'This is the second post.'},
        {'title': 'Third Post', 'content': "I'm fucking disgusted and mad with this fucking bullshit"},
        {'title': 'I AM SO MAD!!!', 'content': 'I AM SO MAD'},
        # More posts...
    ]

    pred_texts = [post['content'] for post in posts]
    # Tokenize texts and create prediction data set
    tokenized_texts = tokenizer(pred_texts, truncation=True, padding=True)
    pred_dataset = Posts(tokenized_texts)

    # Run predictions
    predictions = trainer.predict(pred_dataset)

    # Transform predictions to labels
    preds = predictions.predictions.argmax(-1)
    labels = pd.Series(preds).map(model.config.id2label)
    scores = (np.exp(predictions[0]) / np.exp(predictions[0]).sum(-1, keepdims=True)).max(1)

    print((np.exp(predictions[0]) / np.exp(predictions[0]).sum(-1, keepdims=True)))

    # scores raw
    temp = (np.exp(predictions[0]) / np.exp(predictions[0]).sum(-1, keepdims=True))

    # work in progress
    # container
    anger = []
    disgust = []
    fear = []
    joy = []
    neutral = []
    sadness = []
    surprise = []

    # extract scores (as many entries as exist in pred_texts)
    for i in range(len(pred_texts)):
        anger.append(temp[i][0])
        disgust.append(temp[i][1])
        fear.append(temp[i][2])
        joy.append(temp[i][3])
        neutral.append(temp[i][4])
        sadness.append(temp[i][5])
        surprise.append(temp[i][6])

    print(anger, disgust, fear, joy, neutral, sadness, surprise)

    for post in posts:
        print(post['content'])
        print(classifier(post['content'][0]))
        post['emotion'] = json.dumps(classifier(post['content'][0]))
    return jsonify({'posts': posts})


if __name__ == '__main__':
    app.run(debug=True, port=8000)
