import os
import sys
import getopt
import datetime
import numpy as np
from collections import OrderedDict
from pprint import pprint
import data_loader
import vocabulary_embeddings_extractor
import timeit
import json


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


def detail_model(m):
    mjson = m.to_json()
    with open('model.json', 'w') as fw:
        fw.write(json.dump(mjson))
    m.summary()
    for layer in m.layers:
        print(layer)
        pprint(layer.get_config())
    for input in m.inputs:
        print(input, type(input))
    print(m.outputs)
    print('config:', m.get_config())


def __main__(argv):
    start = timeit.default_timer()

    # # argument parsing and setting default values
    try:
        opts, args = getopt.getopt(argv, '', ['lstm_size=',
                                              'dropout=',
                                              'epochs=',
                                              'padding=',
                                              'batch_size=',
                                              'pre_seed=',
                                              'run=',
                                              'runs=',
                                              'embedding=',
                                              'optimizer=',
                                              'loss=',
                                              'activation1=',
                                              'activation2=',
                                              'vsplit=',
                                              'rich=',
                                              'embeddings='])
    except getopt.GetoptError:
        print('argument error')
        sys.exit(2)

    # optional (and default values)
    options = dict(verbose=1, lstm_size=64, dropout=0.9, epochs=5, padding=100, batch_size=32, pre_seed=12345,
                   run=1, runs=3, embedding='w2v', optimizer='adam', loss='binary_crossentropy',
                   activation1='relu', activation2='sigmoid', vsplit=0.1, rich=3, embeddings=1)

    current_dir = os.getcwd()
    emb_dir = current_dir + '/embedding_caches/'
    embedding_files = dict(w2v="embeddings_cache_file_word2vec.pkl.bz2",
                           d2v="embeddings_cache_file_dict2vec.pkl.bz2",
                           d2v_pf="embeddings_cache_file_dict2vec_prov_freq.pkl.bz2",
                           d2v_pf_lc="embeddings_cache_file_dict2vec_prov_freq_lc.pkl.bz2",
                           d2v_pf_lc2="embeddings_cache_file_dict2vec_prov_freq_lc2.pkl.bz2",
                           ftx="embeddings_cache_file_fastText.pkl.bz2",
                           ftx_pf="embeddings_cache_file_fastText_prov_freq.pkl.bz2",
                           ftx_pf_lc="embeddings_cache_file_fastText_prov_freq_lc.pkl.bz2",
                           ftx_pf_lc2="embeddings_cache_file_fastText_prov_freq_lc2.pkl.bz2")

    for opt, arg in opts:
        if opt == '--lstm_size':
            options['lstm_size'] = int(arg)
        elif opt == '--dropout':
            options['dropout'] = float(arg)
        elif opt == '--epochs':
            options['epochs'] = int(arg)
        elif opt == '--padding':
            options['padding'] = int(arg)
        elif opt == '--batch_size':
            options['batch_size'] = int(arg)
        elif opt == '--pre_seed':
            options['pre_seed'] = int(arg)
        elif opt == '--run':
            options['run'] = int(arg)
        elif opt == '--runs':
            options['runs'] = int(arg)
        elif opt == '--embedding':
            options['embedding'] = arg
        elif opt == '--optimizer':
            options['optimizer'] = arg
        elif opt == '--loss':
            options['loss'] = arg
        elif opt == '--activation1':
            options['activation1'] = arg
        elif opt == '--activation2':
            options['activation2'] = arg
        elif opt == '--vsplit':
            options['vsplit'] = float(arg)
        elif opt == '--rich':
            options['rich'] = int(arg)
        elif opt == '--embeddings':
            options['embeddings'] = int(arg)

    np.random.seed(options['pre_seed'])  # for reproducibility
    from keras.preprocessing import sequence
    from keras import callbacks
    from models import get_attention_lstm_intra_warrant

    # TODO:
    # max padding length = None
    # validation_data = dev (or dedicated batch)
    # shuffle = False
    # model
    # train with or without swap
    # full embeddings
    # combined embeddings (2x no_of_channels or 2x dimensionality)
    # reduced number of channels
    # spelling / pre-processing
    # backend -> re-write

    print('parameters:', options)

    # loading data
    # print('Loading data...')
    embeddings_cache_file = emb_dir + embedding_files[options['embedding']]
    embeddings_cache_file2 = emb_dir + embedding_files['ftx_pf']

    # load pre-extracted word-to-index maps and pre-filtered Glove embeddings
    word_to_indices_map, word_index_to_embeddings_map = \
        vocabulary_embeddings_extractor.load_cached_vocabulary_and_embeddings(embeddings_cache_file)
    word_to_indices_map2, word_index_to_embeddings_map2 = \
        vocabulary_embeddings_extractor.load_cached_vocabulary_and_embeddings(embeddings_cache_file2)

    # print('word_to_indices_map:')
    # pprint(word_to_indices_map)
    # print('word_index_to_embeddings_map:')
    # pprint(word_index_to_embeddings_map)

    # loads data and replaces words with indices from embedding cache
    (train_instance_id_list, train_warrant0_list, train_warrant1_list, train_correct_label_w0_or_w1_list,
     train_reason_list, train_claim_list, train_debate_meta_data_list) = \
        data_loader.load_single_file(current_dir + '/data/train/train-w-swap-full.txt', word_to_indices_map)
    # print('loaded', train_reason_list)

    (dev_instance_id_list, dev_warrant0_list, dev_warrant1_list, dev_correct_label_w0_or_w1_list,
     dev_reason_list, dev_claim_list, dev_debate_meta_data_list) = \
        data_loader.load_single_file(current_dir + '/data/dev/dev-full.txt', word_to_indices_map)

    # pad all sequences
    (train_warrant0_list, train_warrant1_list, train_reason_list, train_claim_list, train_debate_meta_data_list) = [
        sequence.pad_sequences(x, maxlen=options['padding']) for x in
        (train_warrant0_list, train_warrant1_list, train_reason_list, train_claim_list, train_debate_meta_data_list)]
    # print('padded', train_reason_list)

    (dev_warrant0_list, dev_warrant1_list, dev_reason_list, dev_claim_list, dev_debate_meta_data_list) = [
        sequence.pad_sequences(x, maxlen=options['padding']) for x in
        (dev_warrant0_list, dev_warrant1_list, dev_reason_list, dev_claim_list, dev_debate_meta_data_list)]

    assert train_reason_list.shape == train_warrant0_list.shape == \
           train_warrant1_list.shape == train_claim_list.shape == train_debate_meta_data_list.shape

    # ---------------
    # build model and train
    all_runs_report = []  # list of dict

    dt = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')

    results = OrderedDict([('timestamp', dt),
                           ('runtime', ''),
                           ('embedding', embedding_files[options['embedding']].split('.')[0]),
                           ('embeddings', options['embeddings']),
                           ('vocabulary', len(word_index_to_embeddings_map)),
                           ('words in embeddings', ''),
                           ('dimensionality', len(word_index_to_embeddings_map[0])),
                           ('backend', 'Theano'),  # TODO
                           ('classifier', 'AttentionLSTM'),  # TODO
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
                           ('run', options['run']),
                           ('run1 seed', ''),
                           ('run2 seed', ''),
                           ('run3 seed', ''),
                           ('run4 seed', ''),
                           ('run5 seed', ''),
                           ('run1 acc', ''),
                           ('run2 acc', ''),
                           ('run3 acc', ''),
                           ('run4 acc', ''),
                           ('run5 acc', '')])

    def predict(i, m, write_answer=False):
        # model predictions
        predicted_probabilities_dev = m.predict(
            {'sequence_layer_warrant0_input': dev_warrant0_list,
             'sequence_layer_warrant1_input': dev_warrant1_list,
             'sequence_layer_reason_input': dev_reason_list,
             'sequence_layer_claim_input': dev_claim_list,
             'sequence_layer_debate_input': dev_debate_meta_data_list,
             },
            batch_size=options['batch_size'],
            verbose=1)
        predicted_labels_dev = get_predicted_labels(predicted_probabilities_dev)

        # print(predicted_labels_dev)
        assert isinstance(dev_correct_label_w0_or_w1_list, list)
        assert isinstance(dev_correct_label_w0_or_w1_list[0], int)
        assert len(dev_correct_label_w0_or_w1_list) == len(predicted_labels_dev)

        # update report
        rep = dict()
        rep['gold_labels_dev'] = dev_correct_label_w0_or_w1_list
        rep['predicted_labels_dev'] = predicted_labels_dev
        rep['ids_dev'] = dev_instance_id_list
        good_ids = set()
        wrong_ids = set()
        answer = ''
        for g, p, instance_id in zip(rep['gold_labels_dev'], rep['predicted_labels_dev'], rep['ids_dev']):
            if g == p:
                good_ids.add(instance_id)
            else:
                wrong_ids.add(instance_id)
            answer += instance_id + '\t' + str(p) + '\n'

        # calculate scorer accuracy
        results['run' + str(i) + ' acc'] = rep['acc_dev'] = len(good_ids) / (len(good_ids) + len(wrong_ids))
        print("%.3f\t" % rep['acc_dev'])

        # write answer file
        if write_answer:
            answer = '#id\tcorrectLabelW0orW1\n' + answer
            with open('tmp/answer' + str(i) + '_' + dt + '.txt', 'w') as fw:
                fw.write(answer)

        if False:
            print("\nInstances correct")
            print("Good_ids\t", good_ids)
            print("Wrong_ids\t", wrong_ids)

        return rep

    # Creating a Callback subclass that stores each epoch prediction
    class PredictionHistory(callbacks.Callback):
        def __init__(self, idx, modl):
            self.idx = idx
            self.set_model(modl)

        def on_epoch_end(self, epoch, logs={}):
            predict(self.idx, self.model)

    for run_idx in range(options['run'], options['run'] + options['runs']):

        run_seed = results['run' + str(run_idx) + ' seed'] = options['pre_seed'] + run_idx
        np.random.seed(run_seed)  # for reproducibility

        print("Run: ", run_idx)
        print('seed=' + str(run_seed), 'random int=' + str(np.random.randint(100000)))

        arguments = dict(dropout=options['dropout'],
                         lstm_size=options['lstm_size'],
                         optimizer=options['optimizer'],
                         loss=options['loss'],
                         activation1=options['activation1'],
                         activation2=options['activation2'])

        # double input layers with extra embedding
        if options['embeddings'] > 1:
            print('use 2 embeddings in parallel')
            arguments['embeddings2'] = word_index_to_embeddings_map2

        model = get_attention_lstm_intra_warrant(word_index_to_embeddings_map,
                                                 options['padding'],
                                                 rich_context=options['rich'],
                                                 **arguments)

        # define training callbacks
        cb_epoch_csvlogger = callbacks.CSVLogger(filename='tmp/log_' + dt + '.csv')
        cb_epoch_checkpoint = callbacks.ModelCheckpoint(filepath='tmp/model_' + dt + '_{epoch:02d}.hdf5',
                                                        monitor='val_acc',
                                                        verbose=1,
                                                        save_best_only=True,
                                                        )
        cb_epoch_learningratereducer = callbacks.ReduceLROnPlateau(monitor='val_acc',
                                                                   factor=0.5,
                                                                   patience=3,
                                                                   verbose=1,
                                                                   min_lr=0.0001
                                                                   )
        cb_epoch_stopping = callbacks.EarlyStopping(monitor='val_acc',
                                                    patience=5,
                                                    verbose=1,
                                                    )
        cb_epoch_predictions = PredictionHistory(run_idx, model)

        model.fit(
            {'sequence_layer_warrant0_input': train_warrant0_list,
             'sequence_layer_warrant1_input': train_warrant1_list,
             'sequence_layer_reason_input': train_reason_list,
             'sequence_layer_claim_input': train_claim_list,
             'sequence_layer_debate_input': train_debate_meta_data_list,
             },
            train_correct_label_w0_or_w1_list,
            epochs=options['epochs'],
            batch_size=options['batch_size'],
            verbose=options['verbose'],
            validation_split=options['vsplit'],
            callbacks=[cb_epoch_predictions,
                       cb_epoch_csvlogger,
                       cb_epoch_checkpoint,
                       cb_epoch_learningratereducer,
                       cb_epoch_stopping]
        )

        report = predict(run_idx, model)
        all_runs_report.append(report)
        # model.save('checkpoints/models_' + str(run_idx) + '_' + dt + '.h5')

    # show report
    results['runtime'] = timeit.default_timer() - start
    pprint(results)

    # write report file
    values = list(results.values())
    out = "\t".join(map(str, values)) + "\n"
    with open('tmp/report_' + dt + '.csv', 'w') as fw:
        fw.write(out)


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
    __main__(sys.argv[1:])
