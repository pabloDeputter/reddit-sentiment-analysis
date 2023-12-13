import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer

import numpy as np
import seaborn as sns
import pandas as pd
import praw
import dotenv
import os

# Load environment variables
dotenv.load_dotenv()

# Set up PRAW
reddit = praw.Reddit(client_id=os.getenv('CLIENT_ID'), client_secret=os.getenv('CLIENT_SECRET'),
                     user_agent=os.getenv('USER_AGENT'), username=os.getenv('PRAW_USERNAME'),
                     password=os.getenv('PASSWORD'))

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


def calculate_average_scores(results):
    """
    Calculate the average score for each emotion.
    :param results: Results from the model
    :return: average scores
    """
    emotion_totals = {}
    for post in results:
        for emotion_score in post:
            emotion = emotion_score['label']
            score = emotion_score['score']
            if emotion in emotion_totals:
                emotion_totals[emotion]['total'] += score
                emotion_totals[emotion]['count'] += 1
            else:
                emotion_totals[emotion] = {'total': score, 'count': 1}

    # Calculate the average score for each emotion
    average_scores = {emotion: data['total'] / data['count'] for emotion, data in emotion_totals.items()}
    return average_scores


def heatmap(results, subreddit, num_posts):
    """
    Visualize a heatmap for average emotion scores of a single subreddit.
    :param results: Dictionary of average emotion scores
    """
    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(list(results.items()), columns=['Emotion', 'Average Score']).set_index('Emotion')
    df = df.transpose()

    plt.figure(figsize=(14, 4))
    sns.heatmap(df, annot=True, cmap="YlGnBu", fmt=".2f", annot_kws={"size": 10})
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks([])
    plt.title(f'Average Emotion in subreddit: {subreddit} over {num_posts} posts', fontsize=12)
    plt.xlabel('Emotions', fontsize=10)
    plt.tight_layout()
    plt.show()


def barchart(results, subreddit, num_posts):
    """
    Visualize a bar chart for average emotion scores of a single subreddit.
    :param results: Dictionary of average emotion scores
    """
    emotions = list(results.keys())
    scores = [results[emotion] for emotion in emotions]

    plt.figure(figsize=(15, 6))
    plt.bar(emotions, scores, color='skyblue')
    plt.xlabel('Emotions')
    plt.ylabel('Average Score')
    plt.title(f'Average Emotion in subreddit: {subreddit} over {num_posts} posts')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


def analyse(subreddit, num_posts=1000):
    """
    Analyze the emotions of a subreddit.
    :param subreddit: subreddit name
    :param num_posts: number of posts to return

    :return: average scores
    """
    posts = []
    for post in reddit.subreddit(subreddit).top('year', limit=num_posts):
        posts.append({'title': post.title, 'content': post.selftext})

    pred_texts = [post['content'] for post in posts]
    tokenized_texts = tokenizer(pred_texts, truncation=True, padding=True)
    pred_dataset = Posts(tokenized_texts)
    # Run predictions
    predictions = trainer.predict(pred_dataset)

    # scores raw
    temp = (np.exp(predictions[0]) / np.exp(predictions[0]).sum(-1, keepdims=True))

    result = [[{'label': label, 'score': float(score)} for label, score in zip(model.config.id2label.values(), scores)]
              for scores in temp]

    return calculate_average_scores(result)


if __name__ == '__main__':
    subreddit = 'offmychest'
    num_posts = 250
    results = analyse(subreddit, num_posts)

    barchart(results, subreddit, num_posts)
    heatmap(results, subreddit, num_posts)
    # TODO - piechart
