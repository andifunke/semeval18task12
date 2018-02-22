""" train a baseline SVM model """
import argparse
import six.moves.cPickle as cPickle
from sklearn import svm, preprocessing
from constants import EMB_DIR
from svm_test import test_main, get_xy
import pandas as pd


def get_options():
    parser = argparse.ArgumentParser(description='semeval 2018 task 12')
    parser.add_argument('--shrinking', dest='shrinking', action='store_true')
    parser.add_argument('--no-shrinking', dest='shrinking', action='store_false')
    parser.set_defaults(shrinking=False)
    parser.add_argument('--verbose', dest='verbose', action='store_true')
    parser.add_argument('--no-verbose', dest='verbose', action='store_false')
    parser.set_defaults(verbose=True)
    parser.add_argument('--scale', dest='scale', action='store_true')
    parser.add_argument('--no-scale', dest='scale', action='store_false')
    parser.set_defaults(scale=False)
    parser.add_argument('--lowercase', dest='lowercase', action='store_true')
    parser.add_argument('--no-lowercase', dest='lowercase', action='store_false')
    parser.set_defaults(lowercase=False)
    parser.add_argument('--gensim', dest='gensim', action='store_true')
    parser.add_argument('--no-gensim', dest='gensim', action='store_false')
    parser.set_defaults(gensim=False)
    parser.add_argument('--C', default=1.0, type=float)
    parser.add_argument('--cache_size', default=2000, type=int)
    parser.add_argument('--kernel', default='linear', type=str,
                        choices=['linear', 'poly', 'rbf'])
    parser.add_argument('--embedding_size', default=25, type=int,
                        choices=[12, 25, 50, 100])
    parser.add_argument('--embedding_model', default='sg', type=str,
                        choices=['cb', 'sg'])
    parser.add_argument('--clf_file', default='', type=str)
    parser.add_argument('--wv_file', default='custom_embedding_w2v_hs_iter20_sg_100_lc_wiki', type=str)
    return vars(parser.parse_args())


def train_main(predict=False, proba=False, embedding=None, kernel=None, c=None, scale=None):
    options = get_options()
    kernel = options['kernel'] if kernel is None else kernel
    c = options['C'] if c is None else c
    options['scale'] = options['scale'] if scale is None else scale
    options['wv_file'] = options['wv_file'] if embedding is None else embedding
    options['lowercase'] = '_lc' in embedding

    x, y = get_xy('train', options, swap=True)

    if options['scale']:
        scaler = preprocessing.MinMaxScaler()
        scaler.fit(x)
        x = scaler.transform(x)
        scale_fname = EMB_DIR + options['wv_file'] + '_scale.pickle'
        print('saving scaler to', scale_fname)
        with open(scale_fname, 'wb') as f:
            cPickle.dump(scaler, f)
    else:
        scaler = None

    clf = svm.SVC(C=c, cache_size=options['cache_size'], class_weight=None, kernel=kernel,
                  decision_function_shape='ovr', gamma='auto', random_state=None,
                  shrinking=options['shrinking'], tol=0.001, verbose=options['verbose'], probability=proba)
    print('fit model')
    clf.fit(x, y)

    if predict:
        dev_acc = test_main(clf, options, 'dev', scaler)
        test_acc = test_main(clf, options, 'test', scaler)
        return pd.Series({
            'kernel': kernel,
            'embedding': options['wv_file'],
            'C': c,
            'dev_acc': dev_acc,
            'test_acc': test_acc,
            'dims': options['dims'],
            'scale': options['scale'],
        })


if __name__ == '__main__':
    embeddings = [
        'svm_custom_embedding_wv2_sg_ns_iter20_25',
        'svm_custom_embedding_wv2_sg_ns_iter20_50',
        'svm_custom_embedding_wv2_sg_ns_iter20_100',
        'svm_custom_embedding_wv2_sg_ns_iter20_300',
        'custom_embedding_w2v_hs_iter20_sg_100_lc_wiki',
        'custom_embedding_w2v_hs_iter20_sg_300_lc_wiki',
    ]
    kernels = ['linear', 'poly', 'rbf']
    Cs = [1, 100]

    results = []
    for e in embeddings[:]:
        for k in kernels[:]:
            for C in Cs[:]:
                print('\n----------------------------------------------------------')
                result = train_main(predict=True, proba=False, kernel=k, embedding=e, c=C, scale=False)
                print()
                print(result)
                results.append(result)
                df = pd.DataFrame(results)
                df.to_csv('../out/svm_results_swap.csv', sep='\t')
