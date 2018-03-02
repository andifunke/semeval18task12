import datetime
import json
import sys
from collections import OrderedDict
from time import time

import numpy as np
np.random.seed(12345)

import tensorflow as tf
tf.set_random_seed(12345)

from tensorflow import __version__ as tfv
from theano import __version__ as thv
from keras import __version__ as kv, backend as K, callbacks
from keras.models import Model
import pandas as pd
from sklearn.metrics import accuracy_score
from argument_parser import get_options
from models import get_model
from constants import *
from preprocessing import load_embedding, load_data, pad, split_train_dev_test, add_swap


def detail_model(m, mname, save_weights=False):
    mjson = m.to_json()
    with open(mname + '.json', 'w') as fp:
        json.dump(mjson, fp)
    m.summary()
    for layer in m.layers:
        print(layer)
        # pprint(layer.get_config())
    for inpt in m.inputs:
        print(inpt, type(inpt))
    print(m.outputs)
    print('config:', m.get_config())

    if save_weights:
        weights_list = m.get_weights()
        # print(weights_list)
        wname = mname + '_weights.txt'
        print("saving weights to", wname)
        with open(wname, 'w') as fp:
            fp.write(str(weights_list))


class PredictionReport(callbacks.Callback):
    """callback subclass that stores each epoch prediction"""

    def __init__(self, model, options: dict, results: dict, reports: list, df_dev: pd.DataFrame, df_test: pd.DataFrame):
        self.set_model(model)
        self.options = options
        self.results = results
        self.reports = reports
        self.df_dev = df_dev
        self.df_test = df_test
        self.values_dev = np.array(df_dev[CONTENT].values.T.tolist())
        self.values_test = np.array(df_test[CONTENT].values.T.tolist())
        self.best_epoch = dict(epoch=0, dev_acc=0, val_acc=0, config=None, weights=None)

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = dict()

        options = self.options
        results = self.results
        best_epoch = self.best_epoch

        self.results['val_acc'] = logs['val_acc']
        if options['verbose'] > 1:
            print('\npredict:', end='')

        results['epoch'] = epoch + 1
        results['runtime'] = time() - results['start']
        results['dev_acc'], _ = predict(self.model, self.options, self.values_dev, self.df_dev)
        results['test_acc'], _ = predict(self.model, self.options, self.values_test, self.df_test)
        logs['dev_pred'] = results['dev_acc']
        logs['test_pred'] = results['test_acc']

        if results['dev_acc'] > best_epoch['dev_acc']:
            best_epoch['epoch'] = epoch + 1
            best_epoch['dev_acc'] = results['dev_acc']
            best_epoch['val_acc'] = results['val_acc']
            best_epoch['test_acc'] = results['test_acc']
            best_epoch['config'] = self.model.get_config()
            best_epoch['weights'] = self.model.get_weights()
        print('run {:02d} epoch {:02d} has finished with, loss={:.3f}, val_acc={:.3f}, dev_acc={:.3f}, test_acc={:.3f} | '
              'best epoch: {:02d}, val_acc={:.3f}, dev_acc={:.3f}, test_acc={:.3f}'
              .format(results['run'], epoch + 1, logs['loss'], results['val_acc'], results['dev_acc'], results['test_acc'],
                      best_epoch['epoch'], best_epoch['val_acc'], best_epoch['dev_acc'], best_epoch['test_acc']))

        # add report for epoch to reports list
        values = list(results.values())[:-1]
        self.reports.append("\t".join(map(str, values)))

    def persist(self):
        options = self.options
        best_epoch = self.best_epoch
        run_idx = self.results['run']
        dt = self.results['timestamp']

        # --- get best epoch and its predictions for this run -----------------------------
        best_model = Model.from_config(best_epoch['config'])
        best_model.set_weights(best_epoch['weights'])
        fname = '%s{}_%s_rn%2d_ep%2d_ac%.3f{}' % (options['out_path'], dt, run_idx,
                                                  best_epoch['epoch'], best_epoch['dev_acc'])

        # --- predict dev data with best model and write answer file ----------------------
        acc_dev, best_probabilities_dev = \
            predict(best_model, options, self.values_dev, self.df_dev,
                    epoch=best_epoch['epoch'], dev_acc=best_epoch['dev_acc'],
                    write_answer=True, print_answer=False, set_type='dev')
        # save dev probabilities
        np.save(fname.format('probabilities-dev', ''), best_probabilities_dev)
        print('acc_dev: {:.3f}'.format(acc_dev))

        # --- predict test data with best model and write answer file ----------------------
        acc_test, best_probabilities_test = \
            predict(best_model, options, self.values_test, self.df_test,
                    epoch=best_epoch['epoch'], dev_acc=best_epoch['dev_acc'],
                    write_answer=True, print_answer=False, set_type='tst')
        # save test probabilities
        np.save(fname.format('probabilities-tst', ''), best_probabilities_test)

        # --- save model and metrics and predict test if accuracy above threshold ---------
        if acc_dev > options['threshold']:
            print('saving model')
            best_model.save(fname.format('model', '.hdf5'))
            # save dev probabilities
            np.save(fname.format('probabilities-dev', ''), best_probabilities_dev)

        filename = '{}report_{}.csv'.format(options['out_path'], dt)
        with open(filename, 'a') as fw:
            fw.write('\n'.join(self.reports))


def predict(model, options: dict, values: np.ndarray, df: pd.DataFrame, epoch: int=0, dev_acc: int=0,
            print_answer: bool=False, write_answer: bool=False, set_type: str='dev'):

    probabilities = model.predict(
        x={
            'sequence_layer_input_warrant0': values[0],
            'sequence_layer_input_warrant1': values[1],
            'sequence_layer_input_reason': values[2],
            'sequence_layer_input_claim': values[3],
            'sequence_layer_input_debateTitle': values[4],
            'sequence_layer_input_debateInfo': values[5],
        },
        batch_size=32,
        verbose=0
    )
    y_pred = (probabilities > 0.5)
    y_true = df['correctLabelW0orW1'].values
    assert len(y_true) == len(y_pred)

    # calculate accuracy score
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)

    # generate answer file
    df.loc[:, 'pred'] = 1 * y_pred
    df_answer = df[[ID, 'pred']]
    if write_answer:
        fname = '{}answer-{}_{}_rn{:02d}_ep{:02d}_ac{:.3f}.txt'\
            .format(options['out_path'], set_type, options['dt'], options['run'], epoch, dev_acc)
        df_answer.to_csv(fname, sep='\t', index=False, index_label=[ID, LABEL])
    if print_answer:
        print(df_answer)

    return acc, probabilities


def initialize_results(options: dict):
    options['dt'] = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
    return OrderedDict([
            ('timestamp', options['dt']),
            ('run', options['run']),
            ('epoch', 0),
            ('runtime', ''),
            ('argv', str(sys.argv[1:])),
            ('embedding', options['embedding'].split('.')[0]),
            ('embedding2', options['embedding2'].split('.')[0]),
            ('vocabulary', options['vocabulary']),
            ('words in embeddings', ''),
            ('dimensionality', options['dimension']),
            ('backend', options['backend']),
            ('classifier', options['classifier']),
            ('epochs', options['epochs']),
            ('dropout', options['dropout']),
            ('lstm_size', options['lstm_size']),
            ('padding', options['padding']),
            ('batch_size', options['batch_size']),
            ('optimizer', options['optimizer']),
            ('loss', options['loss']),
            ('activation1', options['activation1']),
            ('activation2', options['activation2']),
            ('vsplit', options['vsplit']),
            ('alt_split', options['alt_split']),
            ('dev_test_ratio', options['dev_test_ratio']),
            ('rich', options['rich']),
            ('pre_seed', options['pre_seed']),
            ('runs', options['runs']),
            ('run seed', ''),
            ('val_acc', ''),
            ('dev_acc', ''),
            ('test_acc', ''),
    ])


def __main__():
    options = get_options()
    options['backend'] = K.backend()

    # --- verbosity ----------------------------------------------------------------------------------------------------
    print('argv:', sys.argv[1:])
    print('Python version:', sys.version)
    print('Keras version:', kv)
    if options['backend'] == 'theano':
        print('Theano version:', thv)
    elif options['backend'] == 'tensorflow':
        print('TensorFlow version:', tfv)

    # --- loading ------------------------------------------------------------------------------------------------------
    embedding = load_embedding(options)

    if options['alt_split']:
        df = load_data(dataset=['train', 'dev', 'test'], lc=options['lowercase'], use_indexes=True, options=options)
        print('pad sequences')
        df[CONTENT] = df[CONTENT].applymap(lambda sequence: pad(sequence, padding_size=options['padding']))
        df_train, df_dev, df_test = split_train_dev_test(df, dev_test_ratio=options['dev_test_ratio'])
        df_train = add_swap(df_train)
    else:
        df = load_data(dataset=['train_swap', 'dev', 'test'], lc=options['lowercase'], use_indexes=True, options=options)
        print('pad sequences')
        df[CONTENT] = df[CONTENT].applymap(lambda sequence: pad(sequence, padding_size=options['padding']))
        print('using default split and swap')
        df_train = df[df['set'] == 'train_swap'].copy(deep=True)
        df_dev = df[df['set'] == 'dev'].copy(deep=True)
        df_test = df[df['set'] == 'test'].copy(deep=True)

    # --- dictionary to collect result metrics -------------------------------------------------------------------------
    results = initialize_results(options)

    # --- preparing loop -----------------------------------------------------------------------------------------------
    # initialize reports list with column headlines
    keys = list(results.keys())
    report_head = "\t".join(map(str, keys))
    reports = [report_head]

    # --- loop -> fit model with different seeds -----------------------------------------------------------------------
    # reproducibility from the 2nd run onwards works only with Theano, not with TensorFlow
    for run_idx in range(options['run'], options['run'] + options['runs']):

        # --- preparation ----------------------------------------------------------------------------------------------
        results['start'] = time()

        # for reproducibility... you can't have enough seeds
        # although there is still some randomness going on in TensorFlow, maybe due to parallelization
        results['run'] = run_idx
        run_seed = results['run seed'] = options['pre_seed'] + run_idx
        np.random.seed(run_seed)
        tf.set_random_seed(run_seed)
        print("Run: ", run_idx)
        print('seed=' + str(run_seed), 'random int=' + str(np.random.randint(100000)))

        # --- initializing model ---------------------------------------------------------------------------------------
        model = get_model(options, embedding)

        # --- training model -------------------------------------------------------------------------------------------
        # define training callbacks
        cb_epoch_stopping = callbacks.EarlyStopping(monitor='dev_pred', mode='max', patience=options['patience'])
        cb_epoch_predictions = PredictionReport(model, options, results, reports, df_dev, df_test)
        train = np.array(df_train[CONTENT].values.T.tolist())

        print('fit model')
        model.fit(
            x={
                'sequence_layer_input_warrant0': train[0],
                'sequence_layer_input_warrant1': train[1],
                'sequence_layer_input_reason': train[2],
                'sequence_layer_input_claim': train[3],
                'sequence_layer_input_debateTitle': train[4],
                'sequence_layer_input_debateInfo': train[5],
            },
            y=df_train['correctLabelW0orW1'],
            epochs=options['epochs'],
            batch_size=options['batch_size'],
            validation_split=options['vsplit'],
            callbacks=[cb_epoch_predictions, cb_epoch_stopping],
            verbose=0,
        )

        print('finished in {:.3f} minutes'.format((time() - results['start']) / 60))

        # --- storing model, predictions, metrics ----------------------------------------------------------------------
        cb_epoch_predictions.persist()

        # reset reports list after writing reports for this run before the next run starts to remove head
        reports = ['']


if __name__ == "__main__":
    # np.set_printoptions(precision=6, threshold=50, edgeitems=3, linewidth=1000, suppress=True, nanstr=None,
    #                     infstr=None, formatter=None)
    __main__()
