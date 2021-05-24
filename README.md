# RNN_model_Twitter_Sentiment

Multi-layer Long Short-Term Memory (LSTM) Recurrent Neural Networks (RNN) for text classification sentiment analysis in Python using TensorFlow.

# Basic Usage
Twitter.py loads the pre-trained model and classifies the sentiment of tweets filtered by given phrase using Twitter API. 

Tweets are filtered by a given phrase, start date and number of most recent tweets. The script removes @ mentions but retains the hashtags words when preparing the text for classification.

Model outputs of 0.5 or above are considered to have a positive sentiment whereas outputs of below 0.5 are labelled as having a negative sentiment.

# Model
The model was trained on open source classified twitter data alongside with review data from Amazon. A total of 1,013,000 entries were used as inputs to train the model with an 80:20 train to validation split. 117,000 entries were used for testing.

A vocabulary size of 100,000 words was used with the maximum input length truncated at 200 words.

The inputs were pre-processed by remove punctuation, tokenised and vectorised with tf.keras.layers.experimental.preprocessing.TextVectorization.

The model uses a bidirectional RNN, increasing the performance as inputs are processed front to back and back to front. The model structure is shown below:

Layer (type)   -              Output Shape       -       Param #     
_________________________________________________________________
embedding (Embedding)    -    (None, 200, 64)      -     6400000   
_________________________________________________________________
bidirectional - (Bidirectional (None, 200, 128)   -       66048     
_________________________________________________________________
bidirectional_1 - (Bidirection (None, 64)        -        41216     
_________________________________________________________________
dense (Dense)          -      (None, 64)      -          4160      
_________________________________________________________________
dropout (Dropout)       -     (None, 64)        -        0         
_________________________________________________________________
dense_1 (Dense)        -      (None, 1)         -        65        

Total params: 6,511,489
Trainable params: 6,511,489
Non-trainable params: 0


Model and traning data not included in repository due to size limitations.
