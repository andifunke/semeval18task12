""" validate the SVM models """
import re
from sklearn.metrics import accuracy_score
from os import listdir
import json
import numpy as np
import gensim.models.word2vec as wv
import pandas as pd
from constants import *
from preprocessing import get_data_strings, pad
from sklearn.model_selection import train_test_split

np.random.seed(0)


def get_vectors(sequence, word_vectors, lowercase, zeros):
    return [
        word_vectors[token.lower() if lowercase else token]
        if token in word_vectors
        else zeros
        for token in sequence
    ]


def add_swap(df: pd.DataFrame):
    print('add swap')
    # upsample the train set with swapped warrant 0<->1
    df_copy = df.copy(deep=True)
    df_copy = df_copy.rename(columns={WARRANT0: WARRANT1, WARRANT1: WARRANT0})
    df_copy[LABEL] ^= 1
    df_swapped = pd.concat([df, df_copy]).sort_index(kind="mergesort").reset_index(drop=True)
    df_swapped = df_swapped[KEYS]
    return df_swapped


def get_xy(df: pd.DataFrame, options, padding_size=54):
    print('get x_y')

    lowercase = options['lowercase']
    print('lowercase', lowercase)
    fname = EMB_DIR + options['wv_file'] + '.vec'
    print('loading embeddings from', fname)
    model = wv.Word2Vec.load(fname)
    word_vectors = model.wv
    options['dims'] = dims = len(word_vectors['a'])
    print('embedding dimensions', dims)
    zeros = np.zeros(dims)

    # pad data
    df[CONTENT] = df[CONTENT].applymap(lambda sequence:
                                       pad(sequence, padding_size=padding_size, padding_symbol='.$.'))
    # replace with word vectors
    df[CONTENT] = df[CONTENT].applymap(lambda sequence: get_vectors(sequence, word_vectors, lowercase, zeros))

    x_list = df[CONTENT_MIN].values.tolist()
    y_list = df[LABEL].values.tolist()
    x = np.asarray(x_list)
    x = np.reshape(x, (x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]), order='C')
    y = np.asarray(y_list, dtype=bool, order='C')
    assert len(x) == len(y)
    return x, y


def split_train_dev_test(df: pd.DataFrame, train_ratio=0.6145, dev_test_ratio=0.416, seed: int=0):
    print('split train dev test')
    np.random.seed(seed)
    train, dev_test = train_test_split(df, test_size=None, train_size=train_ratio)
    dev, test = train_test_split(dev_test, test_size=None, train_size=dev_test_ratio)
    return train, dev, test


def get_data(dataset: [str, list]=None, lowercase: bool=True):
    """ get data """
    print('loading data for', dataset)
    df = get_data_strings(lc=lowercase)
    if isinstance(dataset, str) and dataset in FILES.keys():
        df = df[df['set'] == dataset]
    elif isinstance(dataset, list):
        df = df[df['set'].isin(dataset)]
    else:
        return None
    return df.reset_index(drop=True)


def test_main(clf, options: dict, x: np.ndarray = None, y: np.ndarray = None,
              val_set: str = None, scaler=None, proba: bool = False):
    """ a given directory and/or filename overwrites the defaults """
    if val_set is not None:
        print('\n')
        print('start validating on', val_set)
        # x, y_true = get_data(val_set, options)
    else:
        y_true = y

    # using the scaler only if the data was trained on scaled values
    if scaler is not None:
        print('scaling values')
        x = scaler.transform(x)

    print('predict')
    y_pred = clf.predict(x)
    if proba:
        y_pred_probabilities = clf.predict_proba(x)
        print(y_pred_probabilities)
        np.save(options['data_dir'] + options['wv_file'] + '.predictions', y_pred_probabilities)

    options['pred_acc'] = pred_acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    print('accuracy:', pred_acc)

    return pred_acc


def load_data(o):
    files = [f for f in listdir(o['data_dir']) if re.match(o['wv_file'], f)]

    clf = None
    options = None
    scaler = None
    for fname in files:
        if '_clf.' in fname:
            print('loading clf from', fname)
            clf = pd.read_pickle(o['data_dir'] + fname)
        elif '_scale.' in fname:
            print('loading scaler from', fname)
            scaler = pd.read_pickle(o['data_dir'] + fname)
        elif '_options.' in fname:
            print('loading options from', fname)

            options = json.load(open(o['data_dir'] + fname))

    if clf is None:
        print("no clf file found. exit")
        return

    if options is None:
        print("no options file found. exit")
        return

    test_main(clf, options, scaler)


if __name__ == '__main__':
    from svm_train import get_options

    OPTIONS = get_options()
    load_data(OPTIONS)
