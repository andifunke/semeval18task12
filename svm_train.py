"""
in parts inspired by http://scikit-learn.org/stable/auto_examples/text/document_classification_20newsgroups.html#sphx-glr-auto-examples-text-document-classification-20newsgroups-py
"""
import argparse
import json
from itertools import chain
from os import path
import numpy as np
import six.moves.cPickle as cPickle
from sklearn import svm, preprocessing
from data_analyzer import *


# argument parsing and setting default values
def get_options():
    parser = argparse.ArgumentParser(description='nlp exercise 2')

    parser.add_argument('--prepare', dest='prepare', action='store_true')
    parser.add_argument('--no-prepare', dest='prepare', action='store_false')
    parser.set_defaults(prepare=False)
    parser.add_argument('--train', dest='train', action='store_true')
    parser.add_argument('--no-train', dest='train', action='store_false')
    parser.set_defaults(train=False)
    parser.add_argument('--test', dest='test', action='store_true')
    parser.add_argument('--no-test', dest='test', action='store_false')
    parser.set_defaults(test=False)
    parser.add_argument('--shrinking', dest='shrinking', action='store_true')
    parser.add_argument('--no-shrinking', dest='shrinking', action='store_false')
    parser.set_defaults(shrinking=False)
    parser.add_argument('--verbose', dest='verbose', action='store_true')
    parser.add_argument('--no-verbose', dest='verbose', action='store_false')
    parser.set_defaults(verbose=True)
    parser.add_argument('--pretrained', dest='pretrained', action='store_true')
    parser.add_argument('--no-pretrained', dest='pretrained', action='store_false')
    parser.set_defaults(pretrained=False)
    parser.add_argument('--scale', dest='scale', action='store_true')
    parser.add_argument('--no-scale', dest='scale', action='store_false')
    parser.set_defaults(scale=False)
    parser.add_argument('--lowercase', dest='lowercase', action='store_true')
    parser.add_argument('--no-lowercase', dest='lowercase', action='store_false')
    parser.set_defaults(lowercase=False)
    parser.add_argument('--gensim', dest='gensim', action='store_true')
    parser.add_argument('--no-gensim', dest='gensim', action='store_false')
    parser.set_defaults(gensim=False)

    parser.add_argument('--train_size', default=0, type=int,
                        help='train only on slice of given length')
    parser.add_argument('--test_size', default=0, type=int,
                        help='test only on slice of given length')
    parser.add_argument('--C', default=1.0, type=float)
    parser.add_argument('--cache_size', default=2000, type=int)
    parser.add_argument('--max_iter', default=-1, type=int)
    parser.add_argument('--kernel', default='linear', type=str,
                        choices=['linear', 'poly', 'rbf'])
    parser.add_argument('--embedding_size', default=25, type=int,
                        choices=[12, 25, 50, 100])
    parser.add_argument('--embedding_model', default='sg', type=str,
                        choices=['cb', 'sg'])
    parser.add_argument('--clf_file', default='', type=str)
    parser.add_argument('--wv_file', default='', type=str)
    parser.add_argument('--data_dir', default='svm', type=str)

    return vars(parser.parse_args())


OPTIONS = get_options()


def get_xy(dataset):
    """ embedding_size, embedding_model and lowercase are only applicable, if pretrained is False.
    In this case the custom trained embeddings are used. Values must correspond to existing w2v model files. """

    lowercase = True if OPTIONS['wv_file'][-3:] == '_lc' else False
    fname = path.join(OPTIONS['data_dir'], OPTIONS['wv_file'] + '.vec')

    # gensim cannot be used on hpc
    if 'OPTIONS' in globals() and OPTIONS['gensim']:
        import gensim.models.word2vec as wv
        model = wv.Word2Vec.load(fname)
        word_vectors = model.wv
    else:
        word_vectors = pd.read_pickle(fname + '.pickle')
    print('loading embeddings from', fname)

    print('loading train data')
    X_df = get_data(FILES[dataset], pad=True)[['warrant0', 'warrant1', 'reason', 'claim']]
    tprint(X_df, 10)
    y = get_labels(FILES[dataset]).as_matrix().flatten()

    X = []
    for row in X_df.itertuples():
        w0 = [word_vectors[token.lower() if lowercase else token] for token in row[1]]
        w1 = [word_vectors[token.lower() if lowercase else token] for token in row[2]]
        r = [word_vectors[token.lower() if lowercase else token] for token in row[3]]
        c = [word_vectors[token.lower() if lowercase else token] for token in row[4]]
        X.append(list(chain.from_iterable(w0))
                 + list(chain.from_iterable(w1))
                 + list(chain.from_iterable(r))
                 + list(chain.from_iterable(c))
                 )

    X = np.asarray(X, dtype=float, order='C')
    y = np.asarray(y, dtype=bool, order='C')
    assert len(X) == len(y)
    return X, y


def train_main():
    t0 = time()

    X, y = get_xy('train_swap')

    if OPTIONS['scale']:
        scaler = preprocessing.MinMaxScaler()
        scaler.fit(X)
        X = scaler.transform(X)
        scale_fname = path.join(OPTIONS['data_dir'], OPTIONS['wv_file'] + '_scale.pickle')
        print('saving scaler to', scale_fname)
        with open(scale_fname, 'wb') as f:
            cPickle.dump(scaler, f)

    print(OPTIONS)
    print('start training')
    clf = svm.SVC(C=OPTIONS['C'], cache_size=OPTIONS['cache_size'], class_weight=None, kernel=OPTIONS['kernel'],
                  decision_function_shape='ovr', gamma='auto', max_iter=OPTIONS['max_iter'], random_state=None,
                  shrinking=OPTIONS['shrinking'], tol=0.001, verbose=OPTIONS['verbose'])
    print('\n' + str(clf))

    print('fitting...')
    clf.fit(X, y)

    clf_fname = path.join(OPTIONS['data_dir'], OPTIONS['wv_file'] + '_clf.pickle')
    print('\nsaving clf to', clf_fname)
    with open(clf_fname, 'wb') as f:
        cPickle.dump(clf, f)

    OPTIONS['time_train'] = time() - t0
    options_fname = path.join(OPTIONS['data_dir'], OPTIONS['wv_file'] + '_options.json')
    print('saving options to', options_fname)
    with open(options_fname, 'w') as f:
        json.dump(OPTIONS, f)

    print("done in {:f}s".format(OPTIONS['time_train']))


if __name__ == '__main__':
    train_main()
