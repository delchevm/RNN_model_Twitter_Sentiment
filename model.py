import os

# Dont pring INFO, WARNING, and ERROR messages - to standard errors about gpus' memory allocation log
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
import keras
import pickle
from keras.preprocessing import sequence

dataset_dir = "D:/Python/ClassData_model"
train_dir = dataset_dir + '/train/'
test_dir = dataset_dir + '/test/'

batch_size = 32
seed = 42

# Creates training and validation sets using 80:20 split of the training data 
raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
    train_dir,
    batch_size=batch_size,
    validation_split=0.2,
    subset='training',
    seed=seed)

raw_val_ds = tf.keras.preprocessing.text_dataset_from_directory(
    train_dir,
    batch_size=batch_size,
    validation_split=0.2,
    subset='validation',
    seed=seed)

# Craetes test dataset
raw_test_ds = tf.keras.preprocessing.text_dataset_from_directory(
    test_dir, batch_size=batch_size)

# VOCAB_SIZE = first 100000 most popular words; max word limit per entry 200 (ignore any words after 200)
VOCAB_SIZE = 100000
MAX_SEQUENCE_LENGTH = 200

# Standardize, tokenize, and vectorize the datawith preprocessing.TextVectorization
# Standardization - preprocessing the text to remove punctuation
# Tokenization - splitting strings into tokens by splitting on whitespace
# Vectorization - converting tokens into numbers

int_vectorize_layer = tf.keras.layers.experimental.preprocessing.TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode='int', # for models that take word order into account
    output_sequence_length=MAX_SEQUENCE_LENGTH)

# Make a text-only dataset (without labels); call adapt to build an index of strings to integers
train_text = raw_train_ds.map(lambda text, labels: text)
int_vectorize_layer.adapt(train_text)

#vocab = int_vectorize_layer.get_vocabulary()

# Pickle the vectorised layer 
pickle.dump({'config': int_vectorize_layer.get_config(),
             'weights': int_vectorize_layer.get_weights()}
            , open("TextVector_layer.pkl", "wb"))


###########
"""
# Load the pickeled layer to avoid vectorising again
from_disk = pickle.load(open("TextVector_layer.pkl", "rb"))

new_v = tf.keras.layers.experimental.preprocessing.TextVectorization.from_config(from_disk['config'])
new_v.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))
new_v.set_weights(from_disk['weights'])

#vocab = new_v.get_vocabulary()

#print(np.shape(raw_train_ds))
"""

# Model Start
model = tf.keras.Sequential([                                                                         # Builds the model
    int_vectorize_layer,                                                                              # Encoder - text to tokens
    tf.keras.layers.Embedding(input_dim=len(int_vectorize_layer.get_vocabulary()), output_dim=64),    # Creates vectors from words
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),                   # Bidirectional propagates the input forward and backwards
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),                                          # https://machinelearningmastery.com/return-sequences-and-return-states-for-lstms-in-keras/
    tf.keras.layers.Dense(64),                                                                        # Densifies
    tf.keras.layers.Dropout(0.5),                                                                     # Randomly sets input units to 0 and scales others so sum over all inputs is unchanged
    tf.keras.layers.Dense(1, activation="sigmoid")                                                    # Densifies using sigmoid (0 to 1) acctivation
])

# Cpmpile the model specifying loss and optimiser functions
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),  
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])

# Train the model
history = model.fit(raw_train_ds, epochs=10,
                    validation_data=raw_val_ds,
                    validation_steps=30)

test_loss, test_acc = model.evaluate(raw_test_ds)

print('Test Loss:', test_loss)
print('Test Accuracy:', test_acc)

# Model End


model.save("RNN_model.tf")
model.summary()





