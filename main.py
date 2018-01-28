import numpy as np
np.random.seed(12345)
import datetime
import timeit
import json
import sys
from collections import OrderedDict
from pprint import pprint
import data_loader
import vocabulary_embeddings_extractor
from argument_parser import get_options, FILES
from classes import Data

CURRENT_PRED_ACC = 0.0


def get_predicted_labels(predicted_probabilities):
    """
    Converts predicted probability/ies to label(s)
    @param predicted_probabilities: output of the classifier
    @return: labels as integers
    """
    assert isinstance(predicted_probabilities, np.ndarray)

    # if the output vector is a probability distribution, return the maximum value; otherwise
    # it's sigmoid, so 1 or 0
    if predicted_probabilities.shape[-1] > 1:
        predicted_labels_numpy = predicted_probabilities.argmax(axis=-1)
    else:
        predicted_labels_numpy = np.array([1 if p > 0.5 else 0 for p in predicted_probabilities])

    # check type
    assert isinstance(predicted_labels_numpy, np.ndarray)
    # convert to a Python list of integers
    predicted_labels = predicted_labels_numpy.tolist()
    assert isinstance(predicted_labels, list)
    assert isinstance(predicted_labels[0], int)
    # check it matches the gold labels

    return predicted_labels


def detail_model(m, mname, save_weights=False):
    mjson = m.to_json()
    with open(mname + '.json', 'w') as fp:
        json.dump(mjson, fp)
    m.summary()
    for layer in m.layers:
        print(layer)
        pprint(layer.get_config())
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


def predict_test(run_idx, model, data, o, write_answer=False, print_answer=False, epoch=0, pred_acc=0):
    # TODO: generalize for dev and test
    predicted_probabilities = model.predict(
        {
            'sequence_layer_input_warrant0': data.test_warrant0,
            'sequence_layer_input_warrant1': data.test_warrant1,
            'sequence_layer_input_reason': data.test_reason,
            'sequence_layer_input_claim': data.test_claim,
            'sequence_layer_input_debate': data.test_debate,
        },
        batch_size=32,  # options['batch_size'],
        verbose=0
    )

    y_pred = get_predicted_labels(predicted_probabilities)
    # print(predicted_probabilities)
    # print('y_pred:', y_pred)

    ids = data.test_ids
    answer = ''
    for p, instance_id in zip(y_pred, ids):
        answer += instance_id + '\t' + str(p) + '\n'

    answer = '#id\tcorrectLabelW0orW1\n' + answer
    # write answer file
    if write_answer:
        with open('{}answer-tst_{}_rn{:02d}_ep{:02d}_ac{:.3f}.txt'
                  .format(o['out_path'], o['dt'], run_idx, epoch, pred_acc), 'w') as fw:
            fw.write(answer)
    if print_answer:
        print(answer)

    return y_pred, predicted_probabilities


def predict_dev(run_idx, model, data, o, write_answer=False, print_answer=False, epoch=0, pred_acc=0):
    # TODO: generalize for dev and test
    predicted_probabilities = model.predict(
        {
            'sequence_layer_input_warrant0': data.dev_warrant0,
            'sequence_layer_input_warrant1': data.dev_warrant1,
            'sequence_layer_input_reason': data.dev_reason,
            'sequence_layer_input_claim': data.dev_claim,
            'sequence_layer_input_debate': data.dev_debate,
        },
        batch_size=32,  # options['batch_size'],
        verbose=0
    )

    y_true = data.dev_label
    y_pred = get_predicted_labels(predicted_probabilities)
    # print(predicted_probabilities)
    # print('y_true:', y_true)
    # print('y_pred:', y_pred)

    assert isinstance(y_true, list)
    assert isinstance(y_true[0], int)
    assert len(y_true) == len(y_pred)

    ids = data.dev_ids
    good_ids = set()
    wrong_ids = set()
    answer = ''
    for g, p, instance_id in zip(y_true, y_pred, ids):
        if g == p:
            good_ids.add(instance_id)
        else:
            wrong_ids.add(instance_id)
        answer += instance_id + '\t' + str(p) + '\n'

    # calculate scorer accuracy
    acc = len(good_ids) / (len(good_ids) + len(wrong_ids))
    # print("acc = %.3f\t" % acc)

    answer = '#id\tcorrectLabelW0orW1\n' + answer
    # write answer file
    if write_answer and acc > o['threshold']:
        with open('{}answer-dev_{}_rn{:02d}_ep{:02d}_ac{:.3f}.txt'
                  .format(o['out_path'], o['dt'], run_idx, epoch, pred_acc), 'w') as fw:
            fw.write(answer)
    if print_answer:
        print(answer)

    if False:
        print("\nInstances correct")
        print("Good_ids\t", good_ids)
        print("Wrong_ids\t", wrong_ids)

    global CURRENT_PRED_ACC
    CURRENT_PRED_ACC = acc

    return acc, y_pred, predicted_probabilities


def get_embeddings(embedding: str, embeddings_file, emb_dir):
    lc = True if ('_lc' in embedding[-4:]) else False

    if embedding[:2] == 'ce':
        custom = True
        word_to_indices_map, word_index_to_embeddings_map = \
            vocabulary_embeddings_extractor.load_custom_vocabulary_and_embeddings(embeddings_file, emb_dir, lc)
    else:
        custom = False
        # load pre-extracted word-to-index maps and pre-filtered Glove embeddings
        word_to_indices_map, word_index_to_embeddings_map = \
            vocabulary_embeddings_extractor.load_cached_vocabulary_and_embeddings(embeddings_file)
    return word_to_indices_map, word_index_to_embeddings_map, lc, custom


def __main__():
    print('argv:', sys.argv[1:])
    start = timeit.default_timer()
    o, emb_files = get_options()
    dt = o['dt'] = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')

    np.random.seed(o['pre_seed'])

    from keras import __version__ as kv, backend as K
    print('Keras version:', kv)
    backend = K.backend()
    if backend == 'theano':
        from theano import __version__ as thv
        print('Theano version:', thv)
    elif backend == 'tensorflow':
        import tensorflow as tf
        tf.set_random_seed(o['pre_seed'])
        from tensorflow import __version__ as tfv
        print('TensorFlow version:', tfv)

    from keras import callbacks
    from keras.preprocessing import sequence
    from keras.models import Model
    from models import get_model

    # Creating a Callback subclass that stores each epoch prediction
    class PredictionReport(callbacks.Callback):

        def __init__(self, idx: int, modl, reports_list: list):
            self.idx = idx
            self.set_model(modl)
            self.reports = reports_list
            self.best_epoch = dict(epoch=0,
                                   pred_acc=0,
                                   val_acc=0,
                                   config=None,
                                   weights=None)

        def on_epoch_end(self, epoch, logs=None):
            if logs is None:
                logs = dict()
            results['val_acc'] = logs['val_acc']
            if o['verbose'] > 1:
                print('\npredict:', end='')
            results['epoch'] = epoch + 1
            results['runtime'] = timeit.default_timer() - start
            results['pred_acc'], _, __ = predict_dev(self.idx, self.model, data=d, o=o)
            logs['dev_pred'] = results['pred_acc']
            if results['pred_acc'] > self.best_epoch['pred_acc']:
                self.best_epoch['epoch'] = epoch + 1
                self.best_epoch['pred_acc'] = results['pred_acc']
                self.best_epoch['val_acc'] = results['val_acc']
                self.best_epoch['config'] = self.model.get_config()
                self.best_epoch['weights'] = self.model.get_weights()
            print('run {:02d} epoch {:02d} has finished with, loss={:.3f}, val_acc={:.3f}, pred_acc={:.3f} | '
                  'best epoch: {:02d}, val_acc={:.3f}, pred_acc={:.3f}'
                  .format(self.idx, epoch + 1, logs['loss'], results['val_acc'], results['pred_acc'],
                          # logs['dev_pred'],
                          self.best_epoch['epoch'], self.best_epoch['val_acc'], self.best_epoch['pred_acc']))
            if o['verbose'] > 1:
                pprint(results)
            # add report for epoch to reports list
            values = list(results.values())
            self.reports.append("\t".join(map(str, values)))

    # 1st embedding
    # loading data
    np.set_printoptions(precision=6, threshold=50, edgeitems=3, linewidth=1000, suppress=True, nanstr=None,
                           infstr=None, formatter=None)
    embeddings_cache_file = o['code_path'] + o['emb_dir'] + emb_files[o['embedding']]
    word_to_indices_map, word_index_to_embeddings_map, lc, custom = get_embeddings(o['embedding'],
                                                                                   embeddings_cache_file,
                                                                                   o['emb_dir'])
    # print('word_to_indices_map')
    # pprint(word_to_indices_map)
    # print('word_index_to_embeddings_map')
    # pprint(word_index_to_embeddings_map)

    # 2nd embedding ?
    if o['embedding2'] != '':
        embeddings_cache_file2 = o['code_path'] + o['emb_dir'] + emb_files[o['embedding2']]
        word_to_indices_map2, word_index_to_embeddings_map2, lc2, custom2 = get_embeddings(o['embedding2'],
                                                                                           embeddings_cache_file2,
                                                                                           o['emb_dir'])
        print('word_to_indices_map2')
        pprint(word_to_indices_map2)
        print('word_index_to_embeddings_map2')
        pprint(word_index_to_embeddings_map2)
    # quit()

    d = Data()
    # loads data and replaces words with indices from embedding cache
    # train
    print('loading train data')
    (d.train_ids, d.train_warrant0, d.train_warrant1, d.train_label, d.train_reason, d.train_claim, d.train_debate) = \
        data_loader.load_single_file(o['code_path'] + FILES['train_swap'], word_to_indices_map, lc=lc, custom=custom)
    # dev
    print('loading dev data')
    (d.dev_ids, d.dev_warrant0, d.dev_warrant1, d.dev_label, d.dev_reason, d.dev_claim, d.dev_debate) = \
        data_loader.load_single_file(o['code_path'] + FILES['dev'], word_to_indices_map, lc=lc, custom=custom)
    # test
    print('loading test data')
    (d.test_ids, d.test_warrant0, d.test_warrant1, d.test_label, d.test_reason, d.test_claim, d.test_debate) = \
        data_loader.load_single_file(o['code_path'] + FILES['test'], word_to_indices_map,
                                     lc=lc, custom=custom, no_labels=True)

    # pad all sequences
    # train
    (d.train_warrant0, d.train_warrant1, d.train_reason, d.train_claim, d.train_debate) = [
        sequence.pad_sequences(x, maxlen=o['padding']) for x in
        (d.train_warrant0, d.train_warrant1, d.train_reason, d.train_claim, d.train_debate)]
    # dev
    (d.dev_warrant0, d.dev_warrant1, d.dev_reason, d.dev_claim, d.dev_debate) = [
        sequence.pad_sequences(x, maxlen=o['padding']) for x in
        (d.dev_warrant0, d.dev_warrant1, d.dev_reason, d.dev_claim, d.dev_debate)]
    # test
    (d.test_warrant0, d.test_warrant1, d.test_reason, d.test_claim, d.test_debate) = [
        sequence.pad_sequences(x, maxlen=o['padding']) for x in
        (d.test_warrant0, d.test_warrant1, d.test_reason, d.test_claim, d.test_debate)]

    assert d.train_reason.shape == d.train_warrant0.shape == d.train_warrant1.shape == d.train_claim.shape == d.train_debate.shape

    # build model and train
    # all_runs_report = []  # list of dict
    results = OrderedDict(
        [('timestamp', dt),
         ('run', o['run']),
         ('epoch', 0),
         ('runtime', ''),
         ('argv', str(sys.argv[1:])),
         ('embedding', emb_files[o['embedding']].split('.')[0]),
         ('embedding2', o['embedding2']),
         ('vocabulary', len(word_index_to_embeddings_map)),
         ('words in embeddings', ''),
         ('dimensionality', len(word_index_to_embeddings_map[0])),
         ('backend', backend),
         ('classifier', o['classifier']),
         ('epochs', o['epochs']),
         ('dropout', o['dropout']),
         ('lstm_size', o['lstm_size']),
         ('padding', o['padding']),
         ('batch_size', o['batch_size']),
         ('optimizer', o['optimizer']),
         ('loss', o['loss']),
         ('activation1', o['activation1']),
         ('activation2', o['activation2']),
         ('vsplit', o['vsplit']),
         ('rich', o['rich']),
         ('pre_seed', o['pre_seed']),
         ('runs', o['runs']),
         ('run seed', ''),
         ('val_acc', ''),
         ('pred_acc', ''),
         ]
    )

    # initialize reports list with column headlines
    keys = list(results.keys())
    report_head = "\t".join(map(str, keys))
    reports = [report_head]

    # Fit model with different seeds
    for run_idx in range(o['run'], o['run'] + o['runs']):
        start = timeit.default_timer()

        # for reproducibility
        results['run'] = run_idx
        run_seed = results['run seed'] = o['pre_seed'] + run_idx
        np.random.seed(run_seed)
        if backend == 'tensorflow':
            tf.set_random_seed(run_seed)

        global CURRENT_PRED_ACC
        CURRENT_PRED_ACC = 0.0

        print("Run: ", run_idx)
        print('seed=' + str(run_seed), 'random int=' + str(np.random.randint(100000)))

        arguments = dict(
            dropout=o['dropout'],
            lstm_size=o['lstm_size'],
            optimizer=o['optimizer'],
            loss=o['loss'],
            activation1=o['activation1'],
            activation2=o['activation2'],
            dense_factor=o['dense_factor']
        )

        model = get_model(
            o['classifier'],
            word_index_to_embeddings_map,
            o['padding'],
            rich_context=o['rich'],
            **arguments
        )

        # define training callbacks
        cb_epoch_csvlogger = callbacks.CSVLogger(
            filename=o['out_path'] + 'log_' + dt + '.csv'
        )
        cb_epoch_checkpoint = callbacks.ModelCheckpoint(
            filepath=o['out_path'] + 'model_' + dt + '_{epoch:02d}.hdf5',
            monitor='dev_pred',
            verbose=1,
            save_best_only=True,
        )
        cb_epoch_learningratereducer = callbacks.ReduceLROnPlateau(
            monitor='dev_pred',
            factor=0.5,
            patience=max(1, o['patience'] - 2),
            verbose=1,
            min_lr=0.0001
        )
        cb_epoch_stopping = callbacks.EarlyStopping(
            monitor='dev_pred',
            mode='max',
            patience=o['patience'],
            verbose=1,
        )
        cb_epoch_predictions = PredictionReport(run_idx, model, reports)

        cbs = [cb_epoch_predictions,
               # cb_epoch_csvlogger,
               # cb_epoch_learningratereducer,
               cb_epoch_stopping,
               ]
        if o['save_models']:
            cbs.append(cb_epoch_checkpoint)

        model.fit(
            {
                'sequence_layer_input_warrant0': d.train_warrant0,
                'sequence_layer_input_warrant1': d.train_warrant1,
                'sequence_layer_input_reason': d.train_reason,
                'sequence_layer_input_claim': d.train_claim,
                'sequence_layer_input_debate': d.train_debate,
             },
            d.train_label,
            epochs=o['epochs'],
            batch_size=o['batch_size'],
            verbose=0,  # o['verbose'],
            validation_split=o['vsplit'],
            callbacks=cbs,
        )

        print('finished in {:.3f} minutes'.format((timeit.default_timer() - start) / 60))

        # save the best model and its predictions for this run
        try:
            if o['verbose'] == 2:
                print('saving model')
            # restore best model
            best_model = Model.from_config(cb_epoch_predictions.best_epoch['config'])
            best_model.set_weights(cb_epoch_predictions.best_epoch['weights'])
            # save best model
            fname = '%s{}_%s_rn%2d_ep%2d_ac%.3f{}' % (o['out_path'], dt, run_idx,
                                                      cb_epoch_predictions.best_epoch['epoch'],
                                                      cb_epoch_predictions.best_epoch['pred_acc'])
            # predict dev data with best model and write answer file
            acc_dev, best_predictions_dev, best_probabilities_dev = \
                predict_dev(run_idx, best_model, d, o,
                            write_answer=True,
                            print_answer=False,
                            epoch=cb_epoch_predictions.best_epoch['epoch'],
                            pred_acc=cb_epoch_predictions.best_epoch['pred_acc'])
            print('acc_dev: {:.3f}'.format(acc_dev))
            if acc_dev > o['threshold']:
                best_model.save(fname.format('model', '.hdf5'))
                # save dev probabilities
                np.save(fname.format('probabilities-dev', ''), best_probabilities_dev)
                # predict test data with best model and write answer file
                best_predictions_test, best_probabilities_test = \
                    predict_test(run_idx, best_model, d, o,
                                 write_answer=True,
                                 print_answer=False,
                                 epoch=cb_epoch_predictions.best_epoch['epoch'],
                                 pred_acc=cb_epoch_predictions.best_epoch['pred_acc'])
                # save test probabilities
                np.save(fname.format('probabilities-tst', ''), best_probabilities_test)

            filename = '{}report_{}.csv'.format(o['out_path'], dt)
            with open(filename, 'a') as fw:
                fw.write('\n'.join(reports))
            # reset reports list after writing reports for run, but without head
            reports = ['']

        except KeyError:
            sys.stderr.write('KeyError: couldn\'t save model for timestamp={}, '
                             'run={:02d}, epoch={:02d}, accuracy={:.3f}\n'
                             .format(dt,
                                     run_idx,
                                     cb_epoch_predictions.best_epoch['epoch'],
                                     cb_epoch_predictions.best_epoch['pred_acc']))


# 'cheating' metric - not working as expected
def dev_pred(y_true, y_pred):
    from keras import backend as K
    global CURRENT_PRED_ACC
    return K.cast(CURRENT_PRED_ACC, 'float32')


if __name__ == "__main__":
    print("Python version:", sys.version)
    __main__()
