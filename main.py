import os
import sys
import getopt
import datetime
import numpy as np
import pandas as pd
from collections import OrderedDict
from pprint import pprint
import data_loader
import vocabulary_embeddings_extractor
import timeit


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


def __main__(argv):
    start = timeit.default_timer()

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
    test = False

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
    from keras.utils.np_utils import accuracy
    from models import get_attention_lstm_intra_warrant

    # TODO:
    # model
    # full embeddings
    # combined embeddings (2x no_of_channels or 2x dimensionality)
    # reduced number of channels
    # spelling / pre-processing
    # backend -> re-write

    print('parameters:', options)

    # print('Loading data...')
    embeddings_cache_file = emb_dir + embedding_files[options['embedding']]
    embeddings_cache_file2 = emb_dir + embedding_files['ftx_pf']

    # load pre-extracted word-to-index maps and pre-filtered Glove embeddings
    word_to_indices_map, word_index_to_embeddings_map = \
        vocabulary_embeddings_extractor.load_cached_vocabulary_and_embeddings(embeddings_cache_file)
    word_to_indices_map2, word_index_to_embeddings_map2 = \
        vocabulary_embeddings_extractor.load_cached_vocabulary_and_embeddings(embeddings_cache_file2)

    (train_instance_id_list, train_warrant0_list, train_warrant1_list, train_correct_label_w0_or_w1_list,
     train_reason_list, train_claim_list, train_debate_meta_data_list) = \
        data_loader.load_single_file(current_dir + '/data/train-w-swap.tsv', word_to_indices_map)

    (dev_instance_id_list, dev_warrant0_list, dev_warrant1_list, dev_correct_label_w0_or_w1_list,
     dev_reason_list, dev_claim_list, dev_debate_meta_data_list) = \
        data_loader.load_single_file(current_dir + '/data/dev.tsv', word_to_indices_map)

    if test:  # only if test.tsv is available
        (test_instance_id_list, test_warrant0_list, test_warrant1_list, test_correct_label_w0_or_w1_list,
         test_reason_list, test_claim_list, test_debate_meta_data_list) = \
            data_loader.load_single_file(current_dir + '/data/test.tsv', word_to_indices_map)

    # pad all sequences
    (train_warrant0_list, train_warrant1_list, train_reason_list, train_claim_list, train_debate_meta_data_list) = [
        sequence.pad_sequences(x, maxlen=options['padding']) for x in
        (train_warrant0_list, train_warrant1_list, train_reason_list, train_claim_list, train_debate_meta_data_list)]

    if test:  # only if test.tsv is available
        (test_warrant0_list, test_warrant1_list, test_reason_list, test_claim_list, test_debate_meta_data_list) = [
            sequence.pad_sequences(x, maxlen=options['padding']) for x in
            (test_warrant0_list, test_warrant1_list, test_reason_list, test_claim_list, test_debate_meta_data_list)]

    (dev_warrant0_list, dev_warrant1_list, dev_reason_list, dev_claim_list, dev_debate_meta_data_list) = [
        sequence.pad_sequences(x, maxlen=options['padding']) for x in (dev_warrant0_list,
                                                                       dev_warrant1_list,
                                                                       dev_reason_list,
                                                                       dev_claim_list,
                                                                       dev_debate_meta_data_list)]
    assert train_reason_list.shape == train_warrant0_list.shape == \
           train_warrant1_list.shape == train_claim_list.shape == train_debate_meta_data_list.shape

    # ---------------
    all_runs_report = []  # list of dict

    results = OrderedDict([('embedding', embedding_files[options['embedding']].split('.')[0]),
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

    for i in range(options['run'], options['run'] + options['runs']):

        run_seed = results['run' + str(i) + ' seed'] = options['pre_seed'] + i
        np.random.seed(run_seed)  # for reproducibility

        print("Run: ", i)
        print('seed=' + str(run_seed), 'random int=' + str(np.random.randint(100000)))

        # simple bidi-lstm model
        # model = get_attention_lstm(word_index_to_embeddings_map, max_len, rich_context=False, dropout=dropout, lstm_size=lstm_size)
        # simple bidi-lstm model w/ context
        # model = get_attention_lstm(word_index_to_embeddings_map, max_len, rich_context=True, dropout=dropout, lstm_size=lstm_size)
        # intra-warrant attention
        # model = get_attention_lstm_intra_warrant(word_index_to_embeddings_map, max_len, rich_context=False, dropout=dropout, lstm_size=lstm_size)
        # intra-warrant w/ context
        arguments = dict(dropout=options['dropout'],
                         lstm_size=options['lstm_size'],
                         optimizer=options['optimizer'],
                         loss=options['loss'],
                         activation1=options['activation1'],
                         activation2=options['activation2'])
        # double input layers with extra exmbedding
        if options['embeddings'] > 1:
            arguments['embeddings2'] = word_index_to_embeddings_map2

        model = get_attention_lstm_intra_warrant(word_index_to_embeddings_map,
                                                 options['padding'],
                                                 rich_context=options['rich'],
                                                 **arguments)

        model.fit(
            {'sequence_layer_warrant0_input': train_warrant0_list,
             'sequence_layer_warrant1_input': train_warrant1_list,
             'sequence_layer_reason_input': train_reason_list,
             'sequence_layer_claim_input': train_claim_list,
             'sequence_layer_debate_input': train_debate_meta_data_list},
            train_correct_label_w0_or_w1_list,
            nb_epoch=options['epochs'],
            batch_size=options['batch_size'],
            verbose=options['verbose'],
            validation_split=options['vsplit'])

        # model predictions
        predicted_probabilities_dev = model.predict(
            {'sequence_layer_warrant0_input': dev_warrant0_list,
             'sequence_layer_warrant1_input': dev_warrant1_list,
             'sequence_layer_reason_input': dev_reason_list,
             'sequence_layer_claim_input': dev_claim_list,
             'sequence_layer_debate_input': dev_debate_meta_data_list},
            batch_size=options['batch_size'],
            verbose=1)

        if test:
            predicted_probabilities_test = model.predict(
                {'sequence_layer_warrant0_input': test_warrant0_list,
                 'sequence_layer_warrant1_input': test_warrant1_list,
                 'sequence_layer_reason_input': test_reason_list,
                 'sequence_layer_claim_input': test_claim_list,
                 'sequence_layer_debate_input': test_debate_meta_data_list},
                batch_size=options['batch_size'],
                verbose=1)

        predicted_labels_dev = get_predicted_labels(predicted_probabilities_dev)
        # print(predicted_labels_dev)

        if test:
            predicted_labels_test = get_predicted_labels(predicted_probabilities_test)

        assert isinstance(dev_correct_label_w0_or_w1_list, list)
        assert isinstance(dev_correct_label_w0_or_w1_list[0], int)
        assert len(dev_correct_label_w0_or_w1_list) == len(predicted_labels_dev)
        if test:
            assert isinstance(test_correct_label_w0_or_w1_list, list)
            assert isinstance(test_correct_label_w0_or_w1_list[0], int)
            assert len(test_correct_label_w0_or_w1_list) == len(predicted_labels_test)

        acc_dev = accuracy(dev_correct_label_w0_or_w1_list, predicted_labels_dev)
        print('Dev accuracy:', acc_dev)
        if test:
            acc_test = accuracy(test_correct_label_w0_or_w1_list, predicted_labels_test)
            print('Test accuracy:', acc_test)
        # update report
        report = dict()
        results['run' + str(i) + ' acc'] = report['acc_dev'] = acc_dev

        if test:
            report['acc_test'] = acc_test
        report['gold_labels_dev'] = dev_correct_label_w0_or_w1_list
        if test:
            report['gold_labels_test'] = test_correct_label_w0_or_w1_list
        report['predicted_labels_dev'] = predicted_labels_dev
        if test:
            report['predicted_labels_test'] = predicted_labels_test
        report['ids_dev'] = dev_instance_id_list
        if test:
            report['ids_test'] = test_instance_id_list

        all_runs_report.append(report)
        # report_description = description + str(args).replace("\n", " ")
        # finish_report(report, report_description, output_file)

    # show report
    print("Acc dev")
    for r in all_runs_report:
        print("%.3f\t" % r['acc_dev'])
    for r in all_runs_report:
        good_ids = set()
        wrong_ids = set()
        for i, (g, p, instance_id) in enumerate(zip(r['gold_labels_dev'], r['predicted_labels_dev'], r['ids_dev'])):
            if g == p:
                good_ids.add(instance_id)
            else:
                wrong_ids.add(instance_id)
    if False:
        print("\nInstances correct")
        print("Good_ids\t", good_ids)
        print("Wrong_ids\t", wrong_ids)
    if test:
        print("\nAcc test")
        for r in all_runs_report:
            print("%.3f\t" % r['acc_test'], end='')

    sum_ = 0
    for r in all_runs_report:
        sum_ += r['acc_dev']
    results['Avg'] = sum_ / len(all_runs_report)

    pprint(results)
    keys = list(results.keys())
    values = list(results.values())
    out = ''
    # out += "\t".join(keys) + '\n'
    out += "\t".join(map(str, values)) + "\n"
    # out += "\n\nInstances correct"
    # out += "\nGood_ids\t" + str(good_ids)
    # out += "\nWrong_ids\t" + str(wrong_ids)

    # write report file
    dt = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
    with open('tmp/report_' + dt + '.csv', 'w') as fw:
        fw.write(out)

    # write answer file
    for idx, r in enumerate(all_runs_report):
        df = pd.DataFrame({'#id': r['ids_dev'],
                           'correctLabelW0orW1': r['predicted_labels_dev']})
        df.to_csv('tmp/answer_' + str(idx) + '_' + dt + '.txt', sep='\t', index=None)

    stop = timeit.default_timer()
    print('calculated run time:', stop - start)


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
