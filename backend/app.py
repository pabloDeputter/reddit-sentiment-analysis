import json

from flask import Flask, render_template, request, jsonify
import praw
import os
import dotenv
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
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
class Posts(Dataset):
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

@app.route('/api/posts')
def get_posts():
    category = request.args.get('category', 'all')

    posts = [
        {'title': 'First Post', 'content': 'This is the first post.'},
        {'title': 'Second Post', 'content': 'This is the second post.'},
        {'title': 'Third Post', 'content': "I'm fucking disgusted and mad with this fucking bullshit"},
        {'title': 'I AM SO MAD!!!', 'content': 'I AM SO MAD'},
        {'title': 'I AM SO MAD!!!', 'content': '''My Girlfriend's (F32) parents are blaming me (M28) for her suicide and I've started to feel like maybe its my fault.
TW Self Harm
Trigger warning: Suicide

I'm in a really bad frame of mind right now and everything in my head just feels like a haze but I'd try my best to explain what happened.

I met my girlfriend a few years after I finished med school. I didn't know many people in the locality back then as I had just shifted. So I joined a gardening group on the weekends which she was a part of. We came from very different backgrounds but we hit off almost instantly. However, it took over a year of being friends before we decided to be together. Just one random evening when we were sipping coffee at my place, she just looked dead in my eyes and said, "I think I love you". I've repeated that moment in my head every night and it still makes me cry.

The first few months were the best time of my life. We were both shy people individually, but together, we were just wild. We went to karaoke bars and long road trips on weekends to unknown towns. We lived in motels and drunk under the stars. We went rock climbing (which none of us had done before and got so exhausted in the first half an hour). Every month we'd save up money to go to a new fancy restaurant and try a new cuisine. I thought we'd travel the world someday. We had a list of places we'd go, experiences we'd share. On our first anniversary, I made her a catalogue of all our photos with tiny sticky notes about the moment they were taken, along with a wonderful set of plants. They are still in our balcony. She took so much care of them.'''},
        # More posts...
    ]

    pred_texts = [post['content'] for post in posts]
    # Tokenize texts and create prediction data set
    tokenized_texts = tokenizer(pred_texts, truncation=True, padding=True)
    pred_dataset = Posts(tokenized_texts)

    # Run predictions
    predictions = trainer.predict(pred_dataset)

    # scores raw
    temp = (np.exp(predictions[0]) / np.exp(predictions[0]).sum(-1, keepdims=True))

    result = [[{'label': label, 'score': float(score)} for label, score in zip(model.config.id2label.values(), scores)] for scores in temp]

    for post, res in zip(posts, result):
        post['emotion'] = json.dumps([res])
    return jsonify({'posts': posts})


if __name__ == '__main__':
    app.run(debug=True, port=8000)
