import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
import keras
##import tensorflow_datasets as tfds
import pickle
from keras.preprocessing import sequence

dataset_dir = "D:/Python/TestClassData_model"
train_dir = dataset_dir + '/train/'
test_dir = dataset_dir + '/test/'

batch_size = 32
seed = 42

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

raw_test_ds = tf.keras.preprocessing.text_dataset_from_directory(
    test_dir, batch_size=batch_size)

VOCAB_SIZE = 100000
MAX_SEQUENCE_LENGTH = 200

#############

int_vectorize_layer = tf.keras.layers.experimental.preprocessing.TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode='int',
    output_sequence_length=MAX_SEQUENCE_LENGTH)

train_text = raw_train_ds.map(lambda text, labels: text)
int_vectorize_layer.adapt(train_text)

vocab = int_vectorize_layer.get_vocabulary()

pickle.dump({'config': int_vectorize_layer.get_config(),
             'weights': int_vectorize_layer.get_weights()}
            , open("TextVector_layer.pkl", "wb"))


###########
"""
from_disk = pickle.load(open("TextVector_layer.pkl", "rb"))

new_v = tf.keras.layers.experimental.preprocessing.TextVectorization.from_config(from_disk['config'])
new_v.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))
new_v.set_weights(from_disk['weights'])

vocab = new_v.get_vocabulary()

#print(np.shape(raw_train_ds))
"""
#### Model Start

model = tf.keras.Sequential([
    int_vectorize_layer,
    tf.keras.layers.Embedding(input_dim=len(int_vectorize_layer.get_vocabulary()), output_dim=64),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(64),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),  #from_logits=True
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])

history = model.fit(raw_train_ds, epochs=10,
                    validation_data=raw_val_ds,
                    validation_steps=30)

test_loss, test_acc = model.evaluate(raw_test_ds)

print('Test Loss:', test_loss)
print('Test Accuracy:', test_acc)

#### Model End


model.save("RNN_model.tf")
model.summary()





