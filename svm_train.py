"""
in parts inspired by http://scikit-learn.org/stable/auto_examples/text/document_classification_20newsgroups.html#sphx-glr-auto-examples-text-document-classification-20newsgroups-py
"""
import argparse
import json
from os import path
import six.moves.cPickle as cPickle
from sklearn import svm, preprocessing
from data_analyzer import *
from svm_test import test_main, get_xy


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


def train_main(predict=False, proba=False):
    options = get_options()

    t0 = time()

    X, y = get_xy('train_swap', options)

    if options['scale']:
        scaler = preprocessing.MinMaxScaler()
        scaler.fit(X)
        X = scaler.transform(X)
        scale_fname = path.join(options['data_dir'], options['wv_file'] + '_scale.pickle')
        print('saving scaler to', scale_fname)
        with open(scale_fname, 'wb') as f:
            cPickle.dump(scaler, f)
    else:
        scaler = None

    print(options)
    print('start training')
    clf = svm.SVC(C=options['C'], cache_size=options['cache_size'], class_weight=None, kernel=options['kernel'],
                  decision_function_shape='ovr', gamma='auto', max_iter=options['max_iter'], random_state=None,
                  shrinking=options['shrinking'], tol=0.001, verbose=options['verbose'], probability=proba)
    print('\n' + str(clf))

    print('fitting...')
    clf.fit(X, y)

    options['time_train'] = time() - t0

    if predict:
        test_main(clf, options, scaler)
    else:
        clf_fname = path.join(options['data_dir'], options['wv_file'] + '_clf.pickle')
        print('\nsaving clf to', clf_fname)
        with open(clf_fname, 'wb') as f:
            cPickle.dump(clf, f)

        options_fname = path.join(options['data_dir'], options['wv_file'] + '_options.json')
        print('saving options to', options_fname)
        with open(options_fname, 'w') as f:
            json.dump(options, f)

        print("done in {:f}s".format(options['time_train']))


if __name__ == '__main__':
    train_main(predict=True, proba=True)
