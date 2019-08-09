from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
from keras.models import model_from_json
import numpy as np
import random
import sys
import io
import os

class Char_LSTM1():

    text = None
    maxlen = 40
    chars = None
    char_indices = None
    indices_char = None
    model = None
    weightsLoaded = None

    def __init__(self):
        self.weightsLoaded = False
        self.maxlen = 40
        self.model = None

    def BuildModel(self):
        json_file = open(os.path.join(os.getcwd(), 'flaskblog/RNN/aMachado.json'), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(os.path.join(os.getcwd(), 'flaskblog/RNN/aMachado.h5'))
        return loaded_model


    def ResetAttributes(self):
        self.chars = sorted(list(set(self.text)))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))



    def sample(self, preds, temperature=1.0):
        # helper function to sample an index from a probability array
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    def generate_text_logs(self, logs):
        # Function invoked at end of each epoch. Prints generated text.
        print()
        print('----- Generating text after training')

        start_index = random.randint(0, len(self.text) - self.maxlen - 1)
        for diversity in [0.2, 0.5, 1.0, 1.2]:
            print('----- diversity:', diversity)

            generated = ''
            sentence = self.text[start_index: start_index + self.maxlen]
            generated += sentence
            print('----- Generating with seed: "' + sentence + '"')
            sys.stdout.write(generated)

            for i in range(400):
                x_pred = np.zeros((1, self.maxlen, len(self.chars)))
                for t, char in enumerate(sentence):
                    x_pred[0, t, self.char_indices[char]] = 1.

                preds = self.model.predict(x_pred, verbose=0)[0]
                next_index = self.sample(preds, diversity)
                next_char = self.indices_char[next_index]

                generated += next_char
                sentence = sentence[1:] + next_char

                sys.stdout.write(next_char)
                sys.stdout.flush()
            print()

    def generate_text(self, seed, model):
        if (self.model == None):
            self.model = self.BuildModel()
        path='flaskblog/RNN/antoniomachado.txt'
        with io.open(path, encoding='utf-8') as f:
            text = f.read().lower()

        start_index = random.randint(0, len(text) - self.maxlen - 1)
        chars = sorted(list(set(text)))
        char_indices = dict((c, i) for i, c in enumerate(chars))
        indices_char = dict((i, c) for i, c in enumerate(chars))

        for diversity in [0.5, 0.8, 1.0, 1.2]:
            print('----- diversity:', diversity)
            if (len(seed) < self.maxlen - 1):
                sentence = seed + " " + text[start_index: start_index + (self.maxlen - 1 - len(seed))]
            else:
                sentence = seed
            generated = ''

            generated += sentence
            for i in range(400):
                x_pred = np.zeros((1, self.maxlen, 59))
                for t, char in enumerate(sentence):
                    x_pred[0, t, char_indices[char]] = 1.

                preds = self.model.predict(x_pred, verbose=0)[0]
                next_index = self.sample(preds, diversity)
                next_char = indices_char[next_index]

                generated += next_char
                sentence = sentence[1:] + next_char

        poem = [sent for sent in generated.split('\n') if sent != '']
        print(type(poem))
        return poem

    def train_model(self):

        path='antoniomachado.txt'
        with io.open(path, encoding='utf-8') as f:
            text = f.read().lower()
        print('corpus length:', len(text))

        chars = sorted(list(set(text)))
        print('total chars:', len(chars))
        char_indices = dict((c, i) for i, c in enumerate(chars))
        indices_char = dict((i, c) for i, c in enumerate(chars))

        # hyperparameters
        batch_size=128
        epochs=500

        # cut the text in semi-redundant sequences of maxlen characters
        maxlen = 40
        step = 3
        sentences = []
        next_chars = []
        for i in range(0, len(text) - maxlen, step):
            sentences.append(text[i: i + maxlen])
            next_chars.append(text[i + maxlen])
        print('nb sequences:', len(sentences))

        print('Vectorization...')
        x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
        y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
        for i, sentence in enumerate(sentences):
            for t, char in enumerate(sentence):
                x[i, t, char_indices[char]] = 1
            y[i, char_indices[next_chars[i]]] = 1


        # build the model: a single LSTM
        print('Build model...')
        model = Sequential()
        model.add(LSTM(128, input_shape=(maxlen, len(chars))))
        model.add(Dense(len(chars)))
        model.add(Activation('sigmoid'))

        optimizer = RMSprop(lr=0.01)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer)

        #print_callback = LambdaCallback(on_epoch_end=generate_text)
        print_callback = LambdaCallback(on_train_end=generate_text)

        model.fit(x, y,
                batch_size=batch_size,
                epochs=epochs,
                callbacks=[print_callback])

        # serialize model to JSON
        model_json = model.to_json()
        with open("aMachado.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("aMachado.h5")
        print("Saved model to disk")

    def gen_poem(self, seed):

        return self.generate_text(seed, self.model)


    def __del__(self):
        print('\n')
        # print('Destructor llamado')
        # print('\n')