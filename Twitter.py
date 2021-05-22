import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import keras
import tweepy as tw
import string
import re

model = keras.models.load_model("RNN_model.tf")

date_since = "2021-04-18"  #
number_of_tweets = 20

consumer_key= 'xxxxxxxxxx'
consumer_secret= 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
access_token= 'xxxxxxxxxxxxxxxxxxx-xxxxxxxxxxxxxxxxx'
access_token_secret= 'xxxxxxxxxxxxxxxxxxxxxxx'

printable = set(string.printable)

auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True)

tweets = tw.Cursor(api.search, q="edinburgh meadows -filter:retweets",  lang="en", tweet_mode='extended', since=date_since).items(number_of_tweets)

tweets = [[tweet.full_text] for tweet in tweets]

for tweet in tweets:
    tweet = ''.join(tweet)
    filter(lambda x: x in printable, tweet)
    tweet = ''.join(filter(lambda x: x in printable, tweet))
    tweet = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",tweet).split())
    predictions = model.predict(np.array([tweet]))
    print(tweet)
    if predictions[0] >= 0.5:
        print("Positive")
    elif predictions[0] < 0.5:
        print("Negative")

    print(predictions[0])
    print("\n")



