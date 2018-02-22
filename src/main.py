import datetime
import json
import sys
from collections import OrderedDict
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
from preprocessing import get_data_indexes, pad


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

    def __init__(self, model, options: dict, results: dict, reports: list, df: pd.DataFrame):
        self.set_model(model)
        self.options = options
        self.results = results
        self.reports = reports
        self.df = df
        self.best_epoch = dict(epoch=0, pred_acc=0, val_acc=0, config=None, weights=None)

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
        results['runtime'] = datetime.datetime.now() - results['start']
        results['pred_acc'], _ = predict(self.model, self.options, self.df)
        logs['dev_pred'] = results['pred_acc']

        if results['pred_acc'] > best_epoch['pred_acc']:
            best_epoch['epoch'] = epoch + 1
            best_epoch['pred_acc'] = results['pred_acc']
            best_epoch['val_acc'] = results['val_acc']
            best_epoch['config'] = self.model.get_config()
            best_epoch['weights'] = self.model.get_weights()
        print('run {:02d} epoch {:02d} has finished with, loss={:.3f}, val_acc={:.3f}, pred_acc={:.3f} | '
              'best epoch: {:02d}, val_acc={:.3f}, pred_acc={:.3f}'
              .format(results['run'], epoch + 1, logs['loss'], results['val_acc'], results['pred_acc'],
                      best_epoch['epoch'], best_epoch['val_acc'], best_epoch['pred_acc']))

        # add report for epoch to reports list
        del results['start']
        values = list(results.values())
        self.reports.append("\t".join(map(str, values)))

    def persist(self):
        df = self.df
        options = self.options
        best_epoch = self.best_epoch
        run_idx = self.results['run']
        dt = self.results['timestamp']

        # --- get best epoch and its predictions for this run -----------------------------
        best_model = Model.from_config(best_epoch['config'])
        best_model.set_weights(best_epoch['weights'])
        fname = '%s{}_%s_rn%2d_ep%2d_ac%.3f{}' % (options['out_path'], dt, run_idx,
                                                  best_epoch['epoch'], best_epoch['pred_acc'])
        # print('best-fit')
        # print(best_model.to_json())
        # print(best_model.get_config())
        # print(best_model.get_weights())

        # --- predict dev data with best model and write answer file ----------------------
        acc_dev, best_probabilities_dev = \
            predict(best_model, options, df[df.set == 'dev'],
                    epoch=best_epoch['epoch'], pred_acc=best_epoch['pred_acc'],
                    write_answer=True, print_answer=True)
        print('acc_dev: {:.3f}'.format(acc_dev))

        # --- save model and metrics and predict test if accuracy above threshold ---------
        if acc_dev > options['threshold']:
            print('saving model')
            best_model.save(fname.format('model', '.hdf5'))
            # save dev probabilities
            np.save(fname.format('probabilities-dev', ''), best_probabilities_dev)
            # predict test data with best model and write answer file
            acc_test, best_probabilities_test = \
                predict(best_model, options, df[df.set == 'test'],
                        epoch=best_epoch['epoch'], pred_acc=best_epoch['pred_acc'],
                        write_answer=True, print_answer=True)
            # save test probabilities
            np.save(fname.format('probabilities-tst', ''), best_probabilities_test)

        filename = '{}report_{}.csv'.format(options['out_path'], dt)
        with open(filename, 'a') as fw:
            fw.write('\n'.join(self.reports))


def predict(model, options: dict, df: pd.DataFrame, epoch: int=0, pred_acc: int=0,
            print_answer: bool=False, write_answer: bool=False):
    probabilities = model.predict(
        x={
            'sequence_layer_input_warrant0': df['warrant0'].values,
            'sequence_layer_input_warrant1': df['warrant1'].values,
            'sequence_layer_input_reason': df['reason'].values,
            'sequence_layer_input_claim': df['claim'].values,
            'sequence_layer_input_debateTitle': df['debateTitle'].values,
            'sequence_layer_input_debateInfo': df['debateInfo'].values,
        },
        batch_size=32,
        verbose=0
    )
    y_pred = df['predictions'] = (probabilities > 0.5)
    y_true = df['correctLabelW0orW1'].values
    assert len(y_true) == len(y_pred)

    # calculate accuracy score
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)

    # generate answer
    df_answer = df[[ID, 'predictions']]
    if write_answer and acc > options['threshold']:
        fname = '{}answer-dev_{}_rn{:02d}_ep{:02d}_ac{:.3f}.txt'\
            .format(options['out_path'], options['dt'], options['run'], epoch, pred_acc)
        df_answer.to_csv(fname, sep='\t', index=False, index_label=[ID, LABEL])
    if print_answer:
        print(df_answer)

    return acc, probabilities


def get_embeddings(options: dict):
    embedding = options['embedding'].split('.')[0]
    embedding_path = options['code_path'] + options['emb_dir'] + options['embedding'] + '.json'
    lc = True if ('_lc' in embedding) else False

    if embedding[:6] == 'custom':
        if lc:
            freq_path = options['code_path'] + options['emb_dir'] + 'custom_embeddings_freq_lc.json'
        else:
            freq_path = options['code_path'] + options['emb_dir'] + 'custom_embeddings_freq.json'

        # TODO: add OOV and other special vectors
        with open(embedding_path, 'r') as fp:
            print(embedding_path)
            words_to_vectors = json.load(fp)
        with open(freq_path, 'r') as fp:
            print(freq_path)
            words_to_indices = json.load(fp)
        indices_to_vectors = {index: words_to_vectors[word] for word, index in words_to_indices.items()}
        dimension = len(indices_to_vectors[4])
        print(dimension)
        indices_to_vectors[0] = np.asarray([0.0] * dimension)
    else:
        indices_to_vectors = {}

    return indices_to_vectors


def initialize_results(options: dict, indices_to_vectors):
    return OrderedDict([
            ('timestamp', datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')),
            ('run', options['run']),
            ('epoch', 0),
            ('runtime', ''),
            ('argv', str(sys.argv[1:])),
            ('embedding', options['embedding'].split('.')[0]),
            ('embedding2', options['embedding2'].split('.')[0]),
            ('vocabulary', len(indices_to_vectors)),
            ('words in embeddings', ''),
            ('dimensionality', len(indices_to_vectors[0])),
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
            ('rich', options['rich']),
            ('pre_seed', options['pre_seed']),
            ('runs', options['runs']),
            ('run seed', ''),
            ('val_acc', ''),
            ('pred_acc', ''),
        ])


def __main__():
    options = get_options()
    options['backend'] = K.backend()

    # --- verbosity -----------------------------------------------------------
    print('argv:', sys.argv[1:])
    print('Python version:', sys.version)
    print('Keras version:', kv)
    if options['backend'] == 'theano':
        print('Theano version:', thv)
    elif options['backend'] == 'tensorflow':
        print('TensorFlow version:', tfv)

    # --- loading -----------------------------------------------------------
    print('load embedding')
    indices_to_vectors = get_embeddings(options)
    print(indices_to_vectors[0])
    print(indices_to_vectors[1])

    print('load data')
    df = get_data_indexes()

    print('pad sequences')
    df[CONTENT] = df[CONTENT].applymap(lambda sequence: pad(sequence, options['padding']))
    df_train = df[df.set == 'train_swap']

    # --- dictionary to collect result metrics --------------------------------
    results = initialize_results(options, indices_to_vectors)

    # --- preparing loop -----------------------------------------------------
    # initialize reports list with column headlines
    keys = list(results.keys())
    report_head = "\t".join(map(str, keys))
    reports = [report_head]

    # --- loop -> fit model with different seeds -----------------------------
    for run_idx in range(options['run'], options['run'] + options['runs']):
        # --- preparation ----------------------------------------------------
        results['start'] = datetime.datetime.now()
        print(results['start'], type(results['start']))

        # for reproducibility... you can't have enough seeds
        # although there is still some randomness going on in TensorFlow, maybe due to parallelization
        results['run'] = run_idx
        run_seed = results['run seed'] = options['pre_seed'] + run_idx
        np.random.seed(run_seed)
        tf.set_random_seed(run_seed)

        print("Run: ", run_idx)
        print('seed=' + str(run_seed), 'random int=' + str(np.random.randint(100000)))

        # --- initializing model ------------------------------------------------
        print('get model')
        model = get_model(options, indices_to_vectors)
        # print('init model')
        # print(model.to_json())
        # print(model.get_config())
        # print(model.get_weights())

        # --- training model -----------------------------------------------------
        # define training callbacks
        cb_epoch_stopping = callbacks.EarlyStopping(monitor='dev_pred', mode='max', patience=options['patience'])
        cb_epoch_predictions = PredictionReport(model, options, results, reports, df)
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
        # print('post-fit')
        # print(model.to_json())
        # print(model.get_config())
        # print(model.get_weights())

        print('finished in {:.3f} minutes'.format((datetime.datetime.now() - results['start']) / 60))

        # --- storing model, predictions, metrics -------------------------------------
        print('save model')
        cb_epoch_predictions.persist()

        # reset reports list after writing reports for this run before the next run starts to remove head
        reports = ['']


if __name__ == "__main__":
    np.set_printoptions(precision=6, threshold=50, edgeitems=3, linewidth=1000, suppress=True, nanstr=None,
                        infstr=None, formatter=None)
    __main__()
