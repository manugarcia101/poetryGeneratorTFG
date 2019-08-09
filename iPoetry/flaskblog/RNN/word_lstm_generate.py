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
from flaskblog.RNN.utils import *
import random
import csv
import re
import sys
import io
import os

class Word_LSTM():

        Path_Model_Json = None
        Path_Model_H5 = None
        Path = None
        MODEL = None
        maxlen = 20

        def __init__(self):
                self.MODEL = None

        def setWriter(self, writer):
                writer = str(writer)
                writer = writer.replace(" ","")
                writer = writer.lower()
                pmh = str('flaskblog/RNN/'+writer+'word.h5')
                pmj = str('flaskblog/RNN/'+writer+'word.json')
                path = str('flaskblog/RNN/'+writer+'.txt')
                self.Path_Model_H5 = pmh
                self.Path_Model_Json = pmj
                self.Path = path
                self.MODEL = self.BuildModel()
                

        def BuildModel(self):
                json_file = open(os.path.join(os.getcwd(), self.Path_Model_Json), 'r')
                loaded_model_json = json_file.read()
                json_file.close()
                loaded_model = model_from_json(loaded_model_json)
                loaded_model.load_weights(os.path.join(os.getcwd(), self.Path_Model_H5))
                return loaded_model

        def sample(self, preds, temperature=1.0):
                if temperature <= 0:
                        return np.argmax(preds)
                preds = np.asarray(preds).astype('float64')
                preds = np.log(preds) / temperature
                exp_preds = np.exp(preds)
                preds = exp_preds / np.sum(exp_preds)
                probas = np.random.multinomial(1, preds, 1)
                return np.argmax(probas)

        def tokens2text(self, tokens, index_to_word):
                return ' '.join([(index_to_word[i] if i in index_to_word else '') for i in tokens])

        def gen_poem(self, seed):
                
                # path='flaskblog/RNN/duquederivas.txt'
                with io.open(self.Path, encoding='utf-8') as f:
                        text = f.read().lower()
                tokens = tokenize(text)
                words = set(tokens)

                print('Reading embeddings...')
                word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('flaskblog/RNN/glove-sbwc.i25.vec', words)
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

                indices = []
                for t in tokens:
                        if t in word_to_index:
                                indices += [word_to_index[t]]
                tokens = indices

                def_text = ''

                def_text += seed + '\n'
                
                start_index = random.randint(0, len(tokens) - self.maxlen - 1)

                def_text += self.tokens2text(tokens[start_index: start_index + self.maxlen], index_to_word) + '\n'

                for diversity in [0.5, 0.8, 1.0, 1.2]:
                        # La variable sentence es la que se usa como semilla
                        # TODO: Adaptar seed al formato de sentence para usar seed como semilla y no la frase seleccionada al azar
                        sentence = tokens[start_index: start_index + self.maxlen]
                        
                        generated = ''
                        print(self.tokens2text(sentence, index_to_word))
                        print('--')
                        for i in range(60):
                                x_prev = np.zeros((1, self.maxlen))
                                for t, w in enumerate(sentence):
                                        x_prev[0, t] = w
                                
                                preds = self.MODEL.predict(x_prev, verbose=0)
                                next_index = self.sample(preds[0], diversity)
                                next_word = index_to_word[next_index]
                                generated += ' ' + next_word
                                sentence = sentence[1:] + [next_index]
                        def_text += generated
                        def_text += '\n\n'
                return def_text