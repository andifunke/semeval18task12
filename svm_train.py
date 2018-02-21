"""
in parts inspired by http://scikit-learn.org/stable/auto_examples/text/document_classification_20newsgroups.html#sphx-glr-auto-examples-text-document-classification-20newsgroups-py
"""
import argparse
from os import path
import six.moves.cPickle as cPickle
from sklearn import svm, preprocessing
from svm_test import test_main, get_xy
import pandas as pd


def get_options():
    parser = argparse.ArgumentParser(description='semeval 2018 task 12')

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
    parser.add_argument('--wv_file', default='custom_embedding_w2v_hs_iter20_sg_100_lc_wiki', type=str)
    parser.add_argument('--data_dir', default='svm', type=str)

    return vars(parser.parse_args())


def train_main(predict=False, proba=False, embedding=None, kernel=None, C=None):
    options = get_options()
    kernel = options['kernel'] if kernel is None else kernel
    C = options['C'] if C is None else C
    options['wv_file'] = options['wv_file'] if embedding is None else embedding

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

    clf = svm.SVC(C=C, cache_size=options['cache_size'], class_weight=None, kernel=kernel,
                  decision_function_shape='ovr', gamma='auto', random_state=None,
                  shrinking=options['shrinking'], tol=0.001, verbose=options['verbose'], probability=proba)
    print('fit model')
    clf.fit(X, y)

    if predict:
        dev_acc = test_main(clf, options, 'dev', scaler)
        test_acc = test_main(clf, options, 'test', scaler)

    return pd.Series({
        'kernel': kernel,
        'embedding': options['wv_file'],
        'C': C,
        'dev_acc': dev_acc,
        'test_acc': test_acc,
    })


if __name__ == '__main__':
    embeddings = [
        'custom_embedding_w2v_hs_iter20_sg_100_lc_wiki',
        'custom_embedding_w2v_hs_iter20_sg_300_lc_wiki',
        'custom_embedding_wv2_sg_ns_iter20_25',
        'custom_embedding_wv2_sg_ns_iter20_50',
        'custom_embedding_wv2_sg_ns_iter20_100',
        'custom_embedding_wv2_sg_ns_iter20_300',
    ]
    kernels = [
        'linear',
        'poly',
        'rbf',
    ]
    Cs = [1, 1000]
    results = []
    for embedding in embeddings[:]:
        for kernel in kernels[:]:
            for C in Cs:
                print('\n----------------------------------------------------------')
                result = train_main(predict=True, proba=False, kernel=kernel, embedding=embedding, C=C)
                print()
                print(result)
                results.append(result)

                df = pd.DataFrame(results)
                df.to_csv('./svm/results.csv', sep='\t')
