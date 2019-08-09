from __future__ import print_function
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from keras.layers.embeddings import Embedding
from sklearn.metrics import confusion_matrix

def read_glove_vecs(glove_file, valid_vocab=None):
    """
    Read CSV formatted models of Glove embedding_size
    If the 'valid_vocab' parameter is passed, the vocabulary is filtered according to those terms
    """
    with open(glove_file, 'r') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            if valid_vocab and curr_word in valid_vocab:
                words.add(curr_word)
                word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)

        i = 1
        words_to_index = {}
        index_to_words = {}
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
    return words_to_index, index_to_words, word_to_vec_map

def add_new_tokens_one_hot(word_to_index, index_to_word, word_to_vec_map, characters):
    """
    Adds one-hot vectors for additional characters in a word embedding model
    This is a tricky way to add punctuation marks or special characters for only-words based models
    """
    orig_size = word_to_vec_map[index_to_word[1]].shape[0]
    new_size = orig_size + len(set(characters))
    vocab_len = len(word_to_index)
    print('Adding new tokens as one-hot vectors to embeddings model')
    print('Original vocabulary size: ', vocab_len)
    print('Original vectors size: ', orig_size)
    print('New vectors size: ', new_size)

    for (i, c) in enumerate(set(characters)):
        word_to_index[c] = vocab_len + i + 1
        index_to_word[vocab_len + i + 1] = c
    print('New vocabulary size: ', len(word_to_index))
    j = orig_size
    for w in word_to_index.keys():
        if w in set(characters): # on-hot representation for new entries
            word_to_vec_map[w] = np.zeros(new_size)
            word_to_vec_map[w][j] = 1
            j = j + 1
        else:                 # expand with zeros for previous entries
            word_to_vec_map[w] = np.pad(word_to_vec_map[w], (0,len(set(characters))), 'constant', constant_values=(.0))

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def read_csv(filename = 'data/emojify_data.csv'):
    phrase = []
    emoji = []

    with open (filename) as csvDataFile:
        csvReader = csv.reader(csvDataFile)

        for row in csvReader:
            phrase.append(row[0])
            emoji.append(row[1])

    X = np.asarray(phrase)
    Y = np.asarray(emoji, dtype=int)

    return X, Y

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)]
    return Y


def tokenize(sentence):
    """
    Splits words in the sentence and generates a list of tokens with a final <EOF> token
    Words are lower-cased
    """
    tokens = re.compile('(\W)').split(sentence.lower())
    tokens = list(filter(lambda x: x != '' and x != ' ', tokens))
    return tokens

def sentences_to_indices(X, word_to_index, max_len):
    """
    Converts an array of sentences (strings) into an array of indices corresponding to words in the sentences.
    The output shape should be such that it can be given to `Embedding()` (described in Figure 4).

    Arguments:
    X -- array of sentences (strings), of shape (m, 1)
    word_to_index -- a dictionary containing the each word mapped to its index
    max_len -- maximum number of words in a sentence. You can assume every sentence in X is no longer than this.

    Returns:
    X_indices -- array of indices corresponding to words in the sentences from X, of shape (m, max_len)
    """

    m = X.shape[0]                                   # number of training examples

    # Initialize X_indices as a numpy matrix of zeros and the correct shape (≈ 1 line)
    X_indices = np.zeros((m, max_len))

    for i in range(m):                               # loop over training examples

        # Convert the ith training sentence in lower case and split is into words. You should get a list of words.
        sentence_words = tokenize(X[i])

        # Initialize j to 0
        j = 0

        # Loop over the words of sentence_words
        for w in sentence_words:
            # Set the (i,j)th entry of X_indices to the index of the correct word.
            X_indices[i, j] = word_to_index[w]
            # Increment j to j + 1
            j = j + 1

    return X_indices

def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    """
    Creates a Keras Embedding() layer and loads in pre-trained GloVe vectors.

    Arguments:
    word_to_vec_map -- dictionary mapping words to their GloVe vector representation.
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

    Returns:
    embedding_layer -- pretrained layer Keras instance
    """

    vocab_len = len(word_to_index) + 1            # adding 1 to fit Keras embedding (requirement)
    emb_dim = word_to_vec_map["el"].shape[0]      # define dimensionality of your GloVe word vectors

    # Initialize the embedding matrix as a numpy array of zeros of shape (vocab_len, dimensions of word vectors = emb_dim)
    emb_matrix = np.zeros((vocab_len, emb_dim))

    # Set each row "index" of the embedding matrix to be the word vector representation of the "index"th word of the vocabulary
    for word, index in word_to_index.items():
        emb_matrix[index, :] = word_to_vec_map[word]

    # Define Keras embedding layer with the correct output/input sizes, make it trainable. Use Embedding(...). Make sure to set trainable=False.
    embedding_layer = Embedding(input_dim=vocab_len, output_dim=emb_dim, trainable=False)

    # Build the embedding layer, it is required before setting the weights of the embedding layer. Do not modify the "None".
    embedding_layer.build((None,))

    # Set the weights of the embedding layer to the embedding matrix. Your layer is now pretrained.
    embedding_layer.set_weights([emb_matrix])

    return embedding_layer                  

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2))

def analogy(a1, a2, b1):
    ref = word_to_vec_map[b1] - word_to_vec_map[a1] + word_to_vec_map[a2]
    maxsim = 0
    candidate = 'none'
    for w in word_to_vec_map.keys():
        if w != a1 and w != a2 and w != b1:
            wvec = word_to_vec_map[w]
            if wvec.shape == ref.shape:
                sim = cosine_similarity(wvec, ref)
                if sim > maxsim:
                    candidate = w
                    maxsim = sim
                    print (w, sim)
    return candidate
