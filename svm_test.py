"""
in parts inspired by http://scikit-learn.org/stable/auto_examples/text/document_classification_20newsgroups.html#sphx-glr-auto-examples-text-document-classification-20newsgroups-py
"""
import re
from sklearn.metrics import accuracy_score
from os import listdir
import json
from os import path
from itertools import chain
import numpy as np
from data_analyzer import *
import gensim.models.word2vec as wv


def get_vectors(sequence, word_vectors, lowercase, zeros):
    return [
        word_vectors[token.lower() if lowercase else token]
        if token in word_vectors
        else zeros
        for token in sequence
    ]


def get_xy(dataset, options):
    """ embedding_size, embedding_model and lowercase are only applicable, if pretrained is False.
    In this case the custom trained embeddings are used. Values must correspond to existing w2v model files. """

    lowercase = True if '_lc' in options['wv_file'] else False
    print('lowercase', lowercase)
    fname = path.join(options['data_dir'], options['wv_file'] + '.vec')

    print('loading embeddings from', fname)
    model = wv.Word2Vec.load(fname)
    word_vectors = model.wv

    print('loading train data')
    X_df = get_data(FILES[dataset], pad=True)[['warrant0', 'warrant1', 'reason', 'claim']]
    y = get_labels(FILES[dataset]).as_matrix().flatten()

    dims = len(word_vectors['.'])
    print('embedding dimensions', dims)
    zeros = np.zeros(dims)

    X = []
    for row in X_df.itertuples():
        w0 = get_vectors(row[1], word_vectors, lowercase, zeros)
        w1 = get_vectors(row[2], word_vectors, lowercase, zeros)
        r = get_vectors(row[3], word_vectors, lowercase, zeros)
        c = get_vectors(row[4], word_vectors, lowercase, zeros)
        X.append(list(chain.from_iterable(w0))
                 + list(chain.from_iterable(w1))
                 + list(chain.from_iterable(r))
                 + list(chain.from_iterable(c))
                 )
    X = np.asarray(X, dtype=float, order='C')
    y = np.asarray(y, dtype=bool, order='C')
    assert len(X) == len(y)
    return X, y


def test_main(clf, options, val_set='dev', scaler=None, proba=False):
    """ a given directory and/or filename overwrites the defaults """
    print('\n')
    print('start validating on', val_set)

    X, y_true = get_xy(val_set, options)

    # using the scaler only if the data was trained on scaled values
    if scaler is not None:
        print('scaling values')
        X = scaler.transform(X)

    print('predict')
    y_pred = clf.predict(X)
    if proba:
        y_pred_probabilities = clf.predict_proba(X)
        print(y_pred_probabilities)
        np.save(path.join(options['data_dir'], options['wv_file']) + '.predictions', y_pred_probabilities)

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
            clf = pd.read_pickle(path.join(o['data_dir'], fname))
        elif '_scale.' in fname:
            print('loading scaler from', fname)
            scaler = pd.read_pickle(path.join(o['data_dir'], fname))
        elif '_options.' in fname:
            print('loading options from', fname)

            options = json.load(open(path.join(o['data_dir'], fname)))

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
