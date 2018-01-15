import datetime
import timeit
import json
import sys
import six.moves.cPickle as cPickle
from collections import OrderedDict
from pprint import pprint
import numpy as np
from keras.models import load_model

import data_loader
import vocabulary_embeddings_extractor
from argument_parser import get_options
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


def print_maps(word_to_indices_map, word_index_to_embeddings_map):
    print('word_to_indices_map:')
    pprint(word_to_indices_map)
    print('word_index_to_embeddings_map:')
    pprint(word_index_to_embeddings_map)


def detail_model(m):
    mjson = m.to_json()
    with open('model.json', 'w') as fw:
        fw.write(json.dump(mjson))
    m.summary()
    for layer in m.layers:
        print(layer)
        pprint(layer.get_config())
    for inpt in m.inputs:
        print(inpt, type(inpt))
    print(m.outputs)
    print('config:', m.get_config())


def predict(run_idx, model, data, o, write_answer=False, print_answer=False, epoch=0, pred_acc=0):
    # model predictions
    predicted_probabilities_dev = model.predict(
        {'sequence_layer_input_warrant0': data.dev_warrant0_list,
         'sequence_layer_input_warrant1': data.dev_warrant1_list,
         'sequence_layer_input_reason': data.dev_reason_list,
         'sequence_layer_input_claim': data.dev_claim_list,
         'sequence_layer_input_debate': data.dev_debate_meta_data_list,
         },
        batch_size=32,  # options['batch_size'],
        verbose=0)
    # print(predicted_probabilities_dev)
    predicted_labels_dev = get_predicted_labels(predicted_probabilities_dev)

    # print(predicted_labels_dev)
    assert isinstance(data.dev_correct_label_w0_or_w1_list, list)
    assert isinstance(data.dev_correct_label_w0_or_w1_list[0], int)
    assert len(data.dev_correct_label_w0_or_w1_list) == len(predicted_labels_dev)

    # update report
    gold_labels_dev = data.dev_correct_label_w0_or_w1_list
    predicted_labels_dev = predicted_labels_dev
    ids_dev = data.dev_instance_id_list
    good_ids = set()
    wrong_ids = set()
    answer = ''
    for g, p, instance_id in zip(gold_labels_dev, predicted_labels_dev, ids_dev):
        if g == p:
            good_ids.add(instance_id)
        else:
            wrong_ids.add(instance_id)
        answer += instance_id + '\t' + str(p) + '\n'

    # calculate scorer accuracy
    acc_dev = len(good_ids) / (len(good_ids) + len(wrong_ids))
    # print("acc_dev = %.3f\t" % acc_dev)

    answer = '#id\tcorrectLabelW0orW1\n' + answer
    # write answer file
    if write_answer:
        with open('{}answer_{}_rn{:02d}_ep{:02d}_ac{:.3f}.txt'
                          .format(o['out_path'], o['dt'], run_idx, epoch, pred_acc), 'w') as fw:
            fw.write(answer)
    if print_answer:
        print(answer)

    if False:
        print("\nInstances correct")
        print("Good_ids\t", good_ids)
        print("Wrong_ids\t", wrong_ids)

    global CURRENT_PRED_ACC
    CURRENT_PRED_ACC = acc_dev

    return acc_dev, predicted_labels_dev, predicted_probabilities_dev


def __main__():
    print('argv:', sys.argv[1:])
    start = timeit.default_timer()
    o, emb_files = get_options()
    dt = o['dt'] = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')

    np.random.seed(o['pre_seed'])  # for reproducibility
    from keras.preprocessing import sequence
    from keras.models import Sequential, Model
    from keras import callbacks, __version__ as kv, backend as K
    from models import get_model
    print('Keras version:', kv)
    backend = K.backend()
    if backend == 'theano':
        from theano import __version__ as thv
        print('Theano version:', thv)
    elif backend == 'tensorflow':
        from tensorflow import __version__ as tfv
        print('TensorFlow version:', tfv)

    # Creating a Callback subclass that stores each epoch prediction
    class PredictionReport(callbacks.Callback):
        head_written = False

        def __init__(self, idx, modl):
            self.idx = idx
            self.set_model(modl)
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
            results['pred_acc'], _, __ = predict(self.idx, self.model, data=d, o=o)
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
            # write report file
            values = list(results.values())
            out = "\t".join(map(str, values)) + "\n"
            # write headline before first data-line
            if not PredictionReport.head_written:
                keys = list(results.keys())
                out = "\t".join(map(str, keys)) + "\n" + out
                PredictionReport.head_written = True
            filename = '{}report_{}.csv'.format(o['out_path'], dt)
            with open(filename, 'a') as fw:
                fw.write(out)

    # 1st embedding
    # loading data
    embeddings_cache_file = o['code_path'] + o['emb_dir'] + emb_files[o['embedding']]
    lc = True if ('_lc' in o['embedding'][-4:]) and (o['true_lc']) else False
    if lc:
        print('lowercase')

    if o['embedding'][:2] == 'ce':
        # word_vectors = json.load(open(embeddings_cache_file + '.json', 'r', encoding='utf8'))
        # word_vectors = {k: np.array(v) for k, v in word_vectors.items()}
        word_vectors = cPickle.load(open(embeddings_cache_file + '.pickle', 'rb'))
        # print('loading embeddings from', embeddings_cache_file)
        wv_list = sorted(word_vectors.items())
        # pprint(wv_list[:10], width=1600, compact=True)
        word_to_indices_map = {item[0]: index for index, item in enumerate(wv_list)}
        word_index_to_embeddings_map = {index: item[1] for index, item in enumerate(wv_list)}
    else:
        # load pre-extracted word-to-index maps and pre-filtered Glove embeddings
        word_to_indices_map, word_index_to_embeddings_map = \
            vocabulary_embeddings_extractor.load_cached_vocabulary_and_embeddings(embeddings_cache_file)

    # print('word_to_indices_map')
    # pprint(sorted(word_to_indices_map.items())[:10], width=1600, compact=True)
    # print('word_index_to_embeddings_map')
    # pprint(sorted(word_index_to_embeddings_map.items())[:10], width=1600, compact=True)

    # 2nd embedding ?
    word_index_to_embeddings_map2 = None
    if o['embedding2'] != '':
        embeddings_cache_file2 = o['code_path'] + o['emb_dir'] + emb_files[o['embedding2']]
        word_to_indices_map2, word_index_to_embeddings_map2 = \
            vocabulary_embeddings_extractor.load_cached_vocabulary_and_embeddings(embeddings_cache_file2)

    d = Data()
    print('loading train data')
    # loads data and replaces words with indices from embedding cache
    # train
    (d.train_instance_id_list, d.train_warrant0_list, d.train_warrant1_list, d.train_correct_label_w0_or_w1_list,
     d.train_reason_list, d.train_claim_list, d.train_debate_meta_data_list) = \
        data_loader.load_single_file(o['code_path'] + 'data/train-w-swap-full_challenge.tsv', word_to_indices_map, lc=lc)
    print('loading dev data')
    # dev
    (d.dev_instance_id_list, d.dev_warrant0_list, d.dev_warrant1_list, d.dev_correct_label_w0_or_w1_list,
     d.dev_reason_list, d.dev_claim_list, d.dev_debate_meta_data_list) = \
        data_loader.load_single_file(o['code_path'] + 'data/dev-full_challenge.tsv', word_to_indices_map, lc=lc)

    # pad all sequences
    # train
    (d.train_warrant0_list, d.train_warrant1_list, d.train_reason_list, d.train_claim_list, d.train_debate_meta_data_list) = [
        sequence.pad_sequences(x, maxlen=o['padding']) for x in
        (d.train_warrant0_list, d.train_warrant1_list, d.train_reason_list, d.train_claim_list, d.train_debate_meta_data_list)]
    # dev
    (d.dev_warrant0_list, d.dev_warrant1_list, d.dev_reason_list, d.dev_claim_list, d.dev_debate_meta_data_list) = [
        sequence.pad_sequences(x, maxlen=o['padding']) for x in
        (d.dev_warrant0_list, d.dev_warrant1_list, d.dev_reason_list, d.dev_claim_list, d.dev_debate_meta_data_list)]

    assert d.train_reason_list.shape == d.train_warrant0_list.shape == \
           d.train_warrant1_list.shape == d.train_claim_list.shape == d.train_debate_meta_data_list.shape

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

    for run_idx in range(o['run'], o['run'] + o['runs']):
        start = timeit.default_timer()
        # t0 = time()

        results['run'] = run_idx
        run_seed = results['run seed'] = o['pre_seed'] + run_idx
        np.random.seed(run_seed)  # for reproducibility
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

        # double input layers with extra embedding
        if word_index_to_embeddings_map2 is not None:
            print('use 2 embeddings in parallel')
            arguments['embeddings2'] = word_index_to_embeddings_map2

        model = get_model(
            o['classifier'],
            word_index_to_embeddings_map,
            o['padding'],
            rich_context=o['rich'],
            **arguments
        )

        # define training callbacks
        cb_epoch_csvlogger = callbacks.CSVLogger(
            filename=o['out_path'] + 'log_' + dt + '.csv')
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
        cb_epoch_predictions = PredictionReport(run_idx, model)

        cbs = [cb_epoch_predictions,
               # cb_epoch_csvlogger,
               # cb_epoch_learningratereducer,
               cb_epoch_stopping,
               ]
        if o['save_models']:
            cbs.append(cb_epoch_checkpoint)

        model.fit(
            {'sequence_layer_input_warrant0': d.train_warrant0_list,
             'sequence_layer_input_warrant1': d.train_warrant1_list,
             'sequence_layer_input_reason': d.train_reason_list,
             'sequence_layer_input_claim': d.train_claim_list,
             'sequence_layer_input_debate': d.train_debate_meta_data_list,
             },
            d.train_correct_label_w0_or_w1_list,
            epochs=o['epochs'],
            batch_size=o['batch_size'],
            verbose=0,  # o['verbose'],
            validation_split=o['vsplit'],
            callbacks=cbs,
        )

        print('finished in {:.3f} minutes'.format((timeit.default_timer() - start) / 60))

        # save the best model for this run
        try:
            if o['verbose'] == 2:
                print('saving model')
            best_model = Model.from_config(cb_epoch_predictions.best_epoch['config'])
            # print(type(best_model))
            # is comilation necessary?
            best_model.compile(loss=o['loss'], optimizer=o['optimizer'], metrics=['accuracy'])
            best_model.set_weights(cb_epoch_predictions.best_epoch['weights'])
            # print(type(best_model))
            # predict best model to write answer file with predicted labels
            acc, best_predictions, best_probabilities = predict(run_idx, best_model, d, o,
                                                                write_answer=True,
                                                                print_answer=True,
                                                                epoch=cb_epoch_predictions.best_epoch['epoch'],
                                                                pred_acc=cb_epoch_predictions.best_epoch['pred_acc'])
            print('acc:', acc)
            fname = '{}model_{}_rn{:02d}_ep{:02d}_ac{:.3f}'.format(o['out_path'], dt,
                                                                   run_idx,
                                                                   cb_epoch_predictions.best_epoch['epoch'],
                                                                   cb_epoch_predictions.best_epoch['pred_acc'])
            best_model.save(fname + '.hdf5')
            # np.save(fname + '.predictions', best_predictions)
            np.save(fname + '.probabilities', best_probabilities)

            print('LOADED MODEL')
            loaded_model = load_model(fname + '.hdf5')
            acc, best_predictions, best_probabilities = predict(run_idx, loaded_model, d, o,
                                                                write_answer=False,
                                                                print_answer=True,
                                                                epoch=cb_epoch_predictions.best_epoch['epoch'],
                                                                pred_acc=cb_epoch_predictions.best_epoch['pred_acc'])
            print('acc:', acc)


        except KeyError:
            sys.stderr.write('KeyError: couldn\'t save model for timestamp={}, '
                             'run={:02d}, epoch={:02d}, accuracy={:.3f}\n'
                             .format(dt,
                                     run_idx,
                                     cb_epoch_predictions.best_epoch['epoch'],
                                     cb_epoch_predictions.best_epoch['pred_acc']))


# cheating metric - not working as expected
def dev_pred(y_true, y_pred):
    from keras import backend as K
    global CURRENT_PRED_ACC
    # print(CURRENT_PRED_ACC)
    # y = K.mean(y_pred)
    # print(y)
    return K.cast(CURRENT_PRED_ACC, 'float32')


def print_error_analysis_dev(ids: set) -> None:
    """
    Prints instances given in the ids parameter; reads data from dev.tsv
    :param ids: ids
    :return: none
    """
    f = open('data/dev.tsv', 'r')
    lines = f.readlines()
    # remove first line with comments
    del lines[0]

    for line in lines:
        split_line = line.split('\t')
        # "#id warrant0 warrant1 correctLabelW0orW1 reason claim debateTitle debateInfo
        assert len(split_line) == 8

        instance_id = split_line[0]
        if instance_id in ids:
            print(line.strip())


if __name__ == "__main__":
    print("Python version:", sys.version)
    __main__()
