import json

from flask import Flask, render_template, request, jsonify
import praw
import os
import dotenv
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification


dotenv.load_dotenv()

app = Flask(__name__)

# Init. PRAW
reddit = praw.Reddit(client_id=os.getenv('CLIENT_ID'), client_secret=os.getenv('CLIENT_SECRET'),
                     user_agent=os.getenv('USER_AGENT'), username=os.getenv('USERNAME'), password=os.getenv('PASSWORD'))

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
    tokenizer = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
    model = AutoModelForSequenceClassification.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)

    posts = [
        {'title': 'First Post', 'content': 'This is the first post.'},
        {'title': 'Second Post', 'content': 'This is the second post.'},
        {'title': 'Third Post', 'content': "I'm fucking disgusted and mad with this fucking bullshit"},
        {'title': 'I AM SO MAD!!!', 'content': 'I AM SO MAD'},
        # More posts...
    ]

    for post in posts:
        print(post['content'])
        print(classifier(post['content'][0]))
        post['emotion'] = json.dumps(classifier(post['content'][0]))
    return jsonify({'posts': posts})


if __name__ == '__main__':
    app.run(debug=True)
