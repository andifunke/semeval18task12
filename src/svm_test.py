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

np.random.seed(0)


def get_vectors(sequence, word_vectors, lowercase, zeros):
    # for token in sequence:
    #     if token not in word_vectors:
    #         print(token)
    return [
        word_vectors[token.lower() if lowercase else token]
        if token in word_vectors
        else zeros
        for token in sequence
    ]


def get_xy(dataset, options, swap=False):
    lowercase = options['lowercase']
    print('lowercase', lowercase)
    fname = EMB_DIR + options['wv_file'] + '.vec'

    print('loading embeddings from', fname)
    model = wv.Word2Vec.load(fname)
    word_vectors = model.wv
    options['dims'] = dims = len(word_vectors['a'])
    print('embedding dimensions', dims)
    zeros = np.zeros(dims)

    print('loading data')
    # get data
    df = get_data_strings(lc=lowercase)
    df = df[df['set'] == dataset]

    if swap:
        # upsample the data set
        df_swap = df.copy(deep=True)
        df_swap = df_swap.rename(columns={WARRANT0: WARRANT1, WARRANT1: WARRANT0})
        df_swap[LABEL] = df_swap[LABEL] * -1 + 1
        df = pd.concat([df, df_swap]).sort_index(kind="mergesort").reset_index(drop=True)

    # pad data
    df[CONTENT] = df[CONTENT].applymap(lambda sequence: pad(sequence, padding_size=54, padding_symbol='.$.'))
    # replace with word vectors
    df[CONTENT] = df[CONTENT].applymap(lambda sequence: get_vectors(sequence, word_vectors, lowercase, zeros))

    x_list = df[CONTENT_MIN].values.tolist()
    y_list = df[LABEL].values.tolist()
    x = np.asarray(x_list)
    x = np.reshape(x, (x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]), order='C')
    y = np.asarray(y_list, dtype=bool, order='C')

    assert len(x) == len(y)
    return x, y


def test_main(clf, options, val_set='dev', scaler=None, proba=False):
    """ a given directory and/or filename overwrites the defaults """
    print('\n')
    print('start validating on', val_set)

    x, y_true = get_xy(val_set, options)

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
