import os
import pickle
import warnings as w

w.simplefilter(action='ignore', category=FutureWarning)

import keras as ks

import click
import numpy as np
from keras.layers import Embedding, Bidirectional, LSTM, Dense
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras_contrib.layers import CRF
from keras_contrib.utils import save_load_utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
VOCAB_SIZE = 4466
MAX_TEXT_LENGTH = 577


def ner_model():
    model = Sequential()
    model.add(
        Embedding(
            input_dim=VOCAB_SIZE,
            output_dim=64,
            input_length=MAX_TEXT_LENGTH,
            trainable=False,
            mask_zero=True
        ))
    model.add(Bidirectional(LSTM(64, return_sequences=True, recurrent_dropout=0.3)))
    model.add(Dense(7))
    crf = CRF(7)
    model.add(crf)
    model.compile(
        optimizer=ks.optimizers.Adadelta(),
        loss=crf.loss_function,
        metrics=[crf.accuracy])
    return model


def load_obj(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def test_sentences(sentences, model, word_tokenizer, index_pos):
    vec = word_tokenizer.texts_to_sequences([sentences])
    vec = pad_sequences(vec, MAX_TEXT_LENGTH)
    ret = model.predict([vec])[0][-len(sentences):]
    ret = np.argmax(ret, axis=1) + 1
    return [index_pos[x] for x in ret]


def pretty_ner(sentences, ner_ret):
    for x, y in zip(sentences, ner_ret):
        print(x, y)


def init():
    model = ner_model()
    word_tokenizer = load_obj('model/word_tokenizer.pickle')
    index_pos = load_obj('model/index_pos.pickle')
    save_load_utils.load_all_weights(model, r'model/ner_250epoch_weights.h5', include_optimizer=False)
    return model, word_tokenizer, index_pos


_model, _word_tokenizer, _index_pos = init()


@click.command()
@click.argument('sentences')
def predict(sentences):
    ret = test_sentences(sentences, _model, _word_tokenizer, _index_pos)
    pretty_ner(sentences, ret)


if __name__ == '__main__':
    predict()
