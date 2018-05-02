import pickle

import keras as ks
import numpy as np
import tensorflow as tf
from keras.layers import Embedding, Bidirectional, LSTM, Dense
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras_contrib.layers import CRF
from keras_contrib.utils import save_load_utils


def save_obj(filename, obj):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def load_obj(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


MAX_TEXT_LENGTH = 577


def test_sentences(sentences):
    vec = word_tokenizer.texts_to_sequences([sentences])
    vec = pad_sequences(vec, MAX_TEXT_LENGTH)
    ret = model.predict([vec])[0][-len(sentences):]
    ret = np.argmax(ret, axis=1) + 1
    return [index_pos[x] for x in ret]


def pretty_ner(sentences, ner_ret):
    for x, y in zip(sentences, ner_ret):
        print(x, y)


if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    model = Sequential()
    model.add(
        Embedding(
            input_dim=4466,
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

    model.summary()

    word_tokenizer = load_obj('model/word_tokenizer.pickle')
    index_pos = load_obj('model/index_pos.pickle')
    save_load_utils.load_all_weights(model, r'model/ner_250epoch_weights.h5', include_optimizer=False)

    to_test = '琪斯美是日本的“东方project”系列弹幕游戏及其衍生作品的登场角色之一。'
    ner_ret = test_sentences(to_test)
    pretty_ner(to_test, ner_ret)
