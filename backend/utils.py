import praw
import os
import dotenv

dotenv.load_dotenv()


def get_dataset(subreddit, limit):
    # Set up PRAW with your credentials
    reddit = praw.Reddit(client_id=os.getenv('CLIENT_ID'), client_secret=os.getenv('CLIENT_SECRET'),
                         user_agent=os.getenv('USER_AGENT'), username=os.getenv('PRAW_USERNAME'),
                         password=os.getenv('PASSWORD'))

    # Choose the subreddit
    subreddit = reddit.subreddit(subreddit)  # Replace 'subreddit_name' with your target subreddit

    # Fetch the top 10 hot posts
    dataset = []
    for post in subreddit.hot(limit=limit):
        dataset.append({'title': post.title, 'content': post.selftext})
    return dataset
