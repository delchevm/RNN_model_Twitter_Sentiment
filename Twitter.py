import os

# Dont pring INFO, WARNING, and ERROR messages - to standard errors about gpus' memory allocation log
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import keras
import tweepy as tw
import string
import re

# Load model
model = keras.models.load_model("RNN_model.tf")

# Select start date, number of tweets and what to look for
date_since = "2021-04-18"  #
number_of_tweets = 20
phrase = "edinburgh meadows"

# Twitter API credentials
consumer_key= 'xxxxxxxxxx'
consumer_secret= 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
access_token= 'xxxxxxxxxxxxxxxxxxx-xxxxxxxxxxxxxxxxx'
access_token_secret= 'xxxxxxxxxxxxxxxxxxxxxxx'


printable = set(string.printable) # Sets of punctuation, digits, ascii chars

auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True)

# Request
tweets = tw.Cursor(api.search, q=phrase + " -filter:retweets",  lang="en", tweet_mode='extended', since=date_since).items(number_of_tweets)

tweets = [[tweet.full_text] for tweet in tweets]

for tweet in tweets:
    # Cleans tweet before prediction
    tweet = ''.join(tweet)
    tweet = ''.join(filter(lambda x: x in printable, tweet)) # Filter non-ascii characers
    tweet = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",tweet).split()) # Removes @
    
    # Predicts and prints tweet and prediction in console
    predictions = model.predict(np.array([tweet]))
    print(tweet)
    if predictions[0] >= 0.5:
        print("Positive")
    elif predictions[0] < 0.5:
        print("Negative")

    print(predictions[0])
    print("\n")



