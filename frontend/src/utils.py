import os
import time
import dotenv
import praw
import requests
import pickle
from cachetools import TTLCache

# Load environment variables
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


token_cache = {
    'token': None,
    'expires_at': None
}


def get_reddit_token():
    """
    Returns a valid Reddit API token. If a cached token is available, it will be returned. Otherwise, a new token
    will be fetched.

    :return: API token
    """
    current_time = time.time()
    # Check if the token is cached and not expired
    if token_cache['token'] and token_cache['expiry'] > current_time:
        return token_cache['token']

    # Fetch a new token
    auth = requests.auth.HTTPBasicAuth(os.environ['CLIENT_ID'], os.environ['CLIENT_SECRET'])
    data = {
        'grant_type': 'password',
        'username': os.environ['USERNAME'],
        'password': os.environ['PASSWORD']
    }
    headers = {'User-Agent': os.environ['USER_AGENT']}
    response = requests.post('https://www.reddit.com/api/v1/access_token', auth=auth, data=data, headers=headers)

    if response.status_code == 200:
        token_info = response.json()
        token_cache['token'] = token_info['access_token']
        # Assuming the token expires in 1 hour (3600 seconds)
        token_cache['expiry'] = current_time + 3600
        return token_info['access_token']
    else:
        raise Exception("Failed to get token")


def get_subreddit_suggestions(query: str):
    """
    Returns a list of subreddit suggestions based on the query parameter.

    :param query: query parameter
    :return: list of subreddit suggestions
    """
    token = get_reddit_token()
    reddit = praw.Reddit(client_id=os.environ['CLIENT_ID'],
                         client_secret=os.environ['CLIENT_SECRET'],
                         user_agent=os.environ['USER_AGENT'],
                         username=os.environ['USERNAME'],
                         password=os.environ['PASSWORD'])

    subreddits = []
    for subreddit in reddit.subreddits.search_by_name(query, include_nsfw=False):
        if not subreddit.over18:
            subreddits.append(subreddit.display_name)

    return subreddits


def save_cache_to_file(cache, filepath):
    """
    Saves the cache to a file.
    """
    with open(filepath, 'wb') as file:
        pickle.dump(cache, file)


def load_cache_from_file(filepath):
    """
    Loads the cache from a file.
    :param filepath:
    :return:
    """
    try:
        with open(filepath, 'rb') as file:
            return pickle.load(file)
    except (FileNotFoundError, EOFError, pickle.UnpicklingError):
        # Cache upto 1000 items, each for 5 minutes
        return TTLCache(maxsize=1000, ttl=300)
