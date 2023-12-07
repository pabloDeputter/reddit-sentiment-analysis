import praw


def main():
    # Set up PRAW with your credentials
    reddit = praw.Reddit(client_id='Ho0YiNH572NQcYCMcM3rGQ',
                         client_secret='jLc5u4v3jH7O8ijnICUyMaPBDnG7qA',
                         user_agent='mozilla:com.example.sentiment-analysis:v1 (by u/def-not-bot-420)',
                         username='def-not-bot-420',
                         password='IrIsCool69')

    # Choose the subreddit
    subreddit = reddit.subreddit('AITAH')  # Replace 'subreddit_name' with your target subreddit

    # Fetch the top 10 hot posts
    for post in subreddit.hot(limit=200):
        print('TITLE:' + post.title)  # Prints the title of each post
        comments = post.comments
        for comment in comments:
            try:
                print(comment.body)
            except:
                pass


if __name__ == '__main__':
    main()
