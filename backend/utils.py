import os
import time

import dotenv
import praw

dotenv.load_dotenv()


def get_dataset(subreddit, limit):
    assert subreddit != ''
    # Set up PRAW with your credentials

    reddit = praw.Reddit(client_id=os.getenv('CLIENT_ID'), client_secret=os.getenv('CLIENT_SECRET'),
                         user_agent=os.getenv('USER_AGENT'), username=os.getenv('PRAW_USERNAME'),
                         password=os.getenv('PASSWORD'))

    # Choose the subreddit
    for _ in range(10):
        try:
            subreddit = reddit.subreddit(subreddit)  # Replace 'subreddit_name' with your target subreddit
            break
        except ValueError as e:
            print(e)
            time.sleep(0.1)
            pass  # some weird error sometimes occurs
            # ValueError: An invalid value was specified for display_name. Check that the argument for the display_name parameter is not empty.

    # Fetch the top 10 hot posts
    dataset = []
    for post in subreddit.hot(limit=limit):
        dataset.append({'title': post.title, 'content': post.selftext})
    return dataset
