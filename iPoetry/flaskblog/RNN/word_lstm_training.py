from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation, Bidirectional
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
from keras.models import model_from_json
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform
from scipy import stats
from utils import *
import random
import csv
import re
import sys
import io
import numpy as np

np.random.seed(1)

# Read training corpus

path='rubendarío.txt'
with io.open(path, encoding='utf-8') as f:
    text = f.read().lower()
tokens = tokenize(text)
words = set(tokens)
print('Corpus stats:', len(text), "characters, ", len(tokens), 'tokens, ', len(words), 'distinct words')

# Read Glove's word embeddings
# There's a problem with these embeddings: spaces and punctuation marks are not within the model.
# One solution is to add a "one-hot" dimension for each new token we may add: [,;:!¡?¿.“”—]

print('Reading embeddings...')
word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('glove-sbwc.i25.vec', words)
vocabulary = sorted(set(word_to_index.keys()))
vocab_size = len(vocabulary)
print('Vocabulary size:', vocab_size, ' items')
embedding_size = word_to_vec_map[index_to_word[1]].shape[0]
print('Embeddings vector size', embedding_size)
add_new_tokens_one_hot(word_to_index, index_to_word, word_to_vec_map, u',;:!¡?¿.“”—\n')
vocabulary = sorted(set(word_to_index.keys()))
vocab_size = len(vocabulary)
print('Vocabulary size:', vocab_size, 'items')
embedding_size = word_to_vec_map[index_to_word[1]].shape[0]
print('Embeddings vector size:', embedding_size)

# hyperparameters
batch_size = 256
epochs = 1000

# cut the text in semi-redundant sequences of maxlen words
maxlen = 20

# convert words to indices, so tokens are just integer numbers

indices = []
for t in tokens:
    if t in word_to_index:
        indices += [word_to_index[t]]
tokens = indices

# create sequences
step = 3
sentences = []
next_words = []
for i in range(0, len(tokens) - maxlen, step):
    sentences.append(tokens[i: i + maxlen])
    next_words.append(tokens[i + maxlen])
print('Nb sequences:', len(sentences))

print('Vectorization...')
X = np.zeros((len(sentences), maxlen), dtype=np.int32)
Y = np.zeros((len(sentences), vocab_size+1), dtype=np.int32)
for i, sentence in enumerate(sentences):
    for t, w in enumerate(sentence):
        X[i, t] = w
    Y[i, next_words[i]] = 1
print('X\n', X[:3])
print('Y\n', Y[:3])


# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(pretrained_embedding_layer(word_to_vec_map, word_to_index))
model.add(Bidirectional(LSTM(128, return_sequences=False, dropout=0.1, recurrent_dropout=0.1)))
model.add(Dense(64, activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(vocab_size+1, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy')
print(model.summary())

def sample(preds, temperature=1.0):
    if temperature <= 0:
        return np.argmax(preds)
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def tokens2text(tokens):
    return ' '.join([index_to_word[i] for i in tokens])

def generate_text(epoch, logs):
    # Function invoked at end of each epoch. Prints generated text.
    print()
    print('-----')

    start_index = random.randint(0, len(tokens) - maxlen - 1)
    for diversity in [0.5, 0.8, 1.0, 1.2]:
        sentence = tokens[start_index: start_index + maxlen]
        generated = ''
        print(tokens2text(sentence))
        print('--')
        for i in range(40):
            x_prev = np.zeros((1, maxlen))
            for t, w in enumerate(sentence):
                x_prev[0, t] = w

            preds = model.predict(x_prev, verbose=0)
            next_index = sample(preds[0], diversity)
            next_word = index_to_word[next_index]
            generated += ' ' + next_word
            sentence = sentence[1:] + [next_index]
        print(generated)

print_callback = LambdaCallback(on_epoch_end=generate_text)

model.fit(X, Y,
          batch_size=batch_size,
          epochs=epochs,
          callbacks=[print_callback])

# serialize model to JSON
model_json = model.to_json()
with open("rubendaríoword.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("rubendaríoword.h5")
print("Saved model to disk")
