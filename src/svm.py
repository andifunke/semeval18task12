"""
Trains a baseline SVM model.
"""

import numpy as np
np.random.seed(0)
import pandas as pd
import six.moves.cPickle as cPickle
from gensim.models import word2vec as wv
from sklearn import svm, preprocessing
from sklearn.metrics import accuracy_score

from .argument_parser import get_options
from .constants import EMB_DIR, CONTENT, CONTENT_MIN, LABEL
from .preprocessing import load_data, add_swap, split_train_dev_test, pad, get_vectors


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


def validate(clf, options: dict, x: np.ndarray = None, y: np.ndarray = None,
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


def train(predict=False, proba=False, embedding=None, kernel=None, c=None, scale=None,
          alternative_split=True):

    options = get_options()
    kernel = options['kernel'] if kernel is None else kernel
    c = options['C'] if c is None else c
    options['scale'] = options['scale'] if scale is None else scale
    options['wv_file'] = options['wv_file'] if embedding is None else embedding
    options['lowercase'] = '_lc' in embedding

    if alternative_split:
        print('use alternative split')
        df = load_data(dataset=['train', 'dev', 'test'], lc=options['lowercase'])
        # get data set splits
        # tprint(df, 10)
        # equal ratio:
        df_train, df_dev, df_test = split_train_dev_test(df, dev_test_ratio=0.5)
        # odd (= default) ratio:
        # df_train, df_dev, df_test = split_train_dev_test(df)
    else:
        print('use default split')
        df_train = load_data('train', lc=options['lowercase'])
        # df_train_swap = get_data('train_swap', lowercase=options['lowercase'])
        df_dev = load_data('dev', lc=options['lowercase'])
        df_test = load_data('test', lc=options['lowercase'])

    # up-sampling the train set
    df_train = add_swap(df_train)
    # get X and y vectors from the splits
    x_train, y_train = get_xy(df_train, options=options)
    x_dev, y_dev = get_xy(df_dev, options=options)
    x_test, y_test = get_xy(df_test, options=options)

    if options['scale']:
        scaler = preprocessing.MinMaxScaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
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
    clf.fit(x_train, y_train)

    if predict:
        print('\nstart validating on', 'dev')
        dev_acc = validate(clf, options, x=x_dev, y=y_dev, scaler=scaler)
        print('\nstart validating on', 'test')
        test_acc = validate(clf, options, x=x_test, y=y_test, scaler=scaler)
        return pd.Series({
            'C': c,
            'dims': options['dims'],
            'embedding': options['wv_file'],
            'kernel': kernel,
            'scale': options['scale'],
            'dev_acc': dev_acc,
            'test_acc': test_acc,
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
    for e in embeddings[:1]:
        for k in kernels[:1]:
            for C in Cs[:1]:
                print('\n----------------------------------------------------------')
                result = train(predict=True, proba=False, kernel=k, embedding=e, c=C, scale=False)
                print()
                print(result)
                results.append(result)
                df_results = pd.DataFrame(results)[['C', 'dims', 'embedding', 'kernel', 'scale', 'dev_acc', 'test_acc']]
                df_results.to_csv('../out/svm_results_alt-split_equal-ratio_new_.csv', sep='\t', index=None)
