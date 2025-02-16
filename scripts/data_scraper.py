import tweepy
import pandas as pd

# Twitter API Bearer Token (v2 authentication)
bearer_token = "AAAAAAAAAAAAAAAAAAAAAGokzQEAAAAApC4YD9poUMf2rsMNDrNkhhas11U%3DrN90qnpwXgxvKpyzARbAnG91C8abvguQBSEpqcf3B3Yai0yZnj"

# Authenticate using the v2 client
client = tweepy.Client(bearer_token=bearer_token)

def scrape_social_media_data(hashtag, platform, max_posts):
    query = f"{hashtag} lang:en -is:retweet"  # Filters: English tweets, exclude retweets
    tweets = []

    # Fetch tweets
    response = client.search_recent_tweets(
        query=query,
        max_results=min(max_posts, 100),
        tweet_fields=["created_at", "public_metrics"]
    )

    if response.data:
        for tweet in response.data:
            tweets.append({
                "Hashtag": hashtag,
                "Platform": platform,
                "Post_Type": "Text",  # Default to Text; update as needed
                "Engagement": tweet.public_metrics["like_count"],
                "Date": tweet.created_at,
            })

    return pd.DataFrame(tweets)

if __name__ == "__main__":
    hashtag = "#example"
    platform = "Twitter"
    data = scrape_social_media_data(hashtag, platform, max_posts=50)
    print(data.head())
