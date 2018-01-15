# save the best model for this run
import numpy as np
from pprint import pprint

from keras.preprocessing import sequence
from sklearn.metrics import accuracy_score

import data_loader
from classes import Data
from data_loader import load_single_file
from keras.models import load_model
from data_analyzer import get_data
import vocabulary_embeddings_extractor
from main import get_predicted_labels
from results_evaluator import load_results, tprint
import six.moves.cPickle as cPickle
from keras import __version__ as kv

if __name__ == '__main__':

    FILES = dict(
        dev='./data/dev-full_challenge.tsv',
        # dev='./data/dev/dev-full.txt',
        test='./data/test/test-only-data.txt',
        train='./data/train/train-full.txt',
        train_swap='./data/train-w-swap-full_challenge.tsv',
        # train_swap='./data/train/train-w-swap-full.txt',
    )
    model_dir = 'out/'

    # model_file = 'model_2017-12-10_01-29-43-821365_rn01_ep15_ac0.712.hdf5'
    model_file = 'model_2018-01-11_02-30-22-312837_rn03_ep07_ac0.655.h5fs'
    old_model = False

    model_ts = model_file[6:32]
    model_rn = int(model_file[35:37])
    model_ep = int(model_file[40:42])

    # load infos about model
    results = load_results('./results/')

    model_info = results[(results.timestamp == model_ts)
                         & (results.run == model_rn)
                         & (results.epoch == model_ep)]
    tprint(model_info)

    # 1st embedding
    # loading data
    embeddings_cache_file = model_info['embedding'].item()
    embeddings_cache_path = './embedding_caches/' + embeddings_cache_file
    lc = True if ('_lc' in embeddings_cache_file[-4:]) else False

    if embeddings_cache_file[:2] == 'ce':
        # word_vectors = json.load(open(embeddings_cache_file + '.json', 'r', encoding='utf8'))
        # word_vectors = {k: np.array(v) for k, v in word_vectors.items()}
        word_vectors = cPickle.load(open(embeddings_cache_path + '.pickle', 'rb'))
        wv_list = sorted(word_vectors.items())
        word_to_indices_map = {item[0]: index for index, item in enumerate(wv_list)}
        word_index_to_embeddings_map = {index: item[1] for index, item in enumerate(wv_list)}
    else:
        # load pre-extracted word-to-index maps and pre-filtered Glove embeddings
        word_to_indices_map, word_index_to_embeddings_map = \
            vocabulary_embeddings_extractor.load_cached_vocabulary_and_embeddings(embeddings_cache_path + '.pkl.bz2')
    pprint(word_to_indices_map)
    np.set_printoptions(precision=6, threshold=50, edgeitems=None, linewidth=6000, suppress=True, nanstr=None,
                     infstr=None, formatter=None)
    pprint(word_index_to_embeddings_map, width=7000)
    quit()

    fpath_model = model_dir + model_file
    model = load_model(fpath_model, compile=False)
    # no need to compile:
    # model.compile(loss=model_info['loss'].item(), optimizer=model_info['optimizer'].item(), metrics=['accuracy'])

    d = Data()
    # loads data and replaces words with indices from embedding cache
    # train
    print('loading train data')
    (d.train_instance_id_list, d.train_warrant0_list, d.train_warrant1_list, d.train_correct_label_w0_or_w1_list,
     d.train_reason_list, d.train_claim_list, d.train_debate_meta_data_list) = \
        data_loader.load_single_file(FILES['train_swap'], word_to_indices_map, lc=lc)
    # dev
    print('loading dev data')
    (d.dev_instance_id_list, d.dev_warrant0_list, d.dev_warrant1_list, d.dev_correct_label_w0_or_w1_list,
     d.dev_reason_list, d.dev_claim_list, d.dev_debate_meta_data_list) = \
        data_loader.load_single_file(FILES['dev'], word_to_indices_map, lc=lc)

    # pad all sequences
    # train
    (d.train_warrant0_list, d.train_warrant1_list, d.train_reason_list, d.train_claim_list, d.train_debate_meta_data_list) = [
        sequence.pad_sequences(x, maxlen=model_info['padding'].item()) for x in
        (d.train_warrant0_list, d.train_warrant1_list, d.train_reason_list, d.train_claim_list, d.train_debate_meta_data_list)]
    # dev
    (d.dev_warrant0_list, d.dev_warrant1_list, d.dev_reason_list, d.dev_claim_list, d.dev_debate_meta_data_list) = [
        sequence.pad_sequences(x, maxlen=model_info['padding'].item()) for x in
        (d.dev_warrant0_list, d.dev_warrant1_list, d.dev_reason_list, d.dev_claim_list, d.dev_debate_meta_data_list)]

    if old_model:
        predicted_probabilities_dev = model.predict(
            {'sequence_layer_warrant0_input': d.dev_warrant0_list,
             'sequence_layer_warrant1_input': d.dev_warrant1_list,
             'sequence_layer_reason_input': d.dev_reason_list,
             'sequence_layer_claim_input': d.dev_claim_list,
             'sequence_layer_debate_input': d.dev_debate_meta_data_list,
             },
        )
    else:
        predicted_probabilities_dev = model.predict(
            {'sequence_layer_input_warrant0': d.dev_warrant0_list,
             'sequence_layer_input_warrant1': d.dev_warrant1_list,
             'sequence_layer_input_reason': d.dev_reason_list,
             'sequence_layer_input_claim': d.dev_claim_list,
             'sequence_layer_input_debate': d.dev_debate_meta_data_list,
             },
        )
    # print(predicted_probabilities_dev)
    ids_dev = d.dev_instance_id_list
    gold_labels_dev = d.dev_correct_label_w0_or_w1_list
    predicted_labels_dev = get_predicted_labels(predicted_probabilities_dev)

    print('Keras version:', kv)
    tprint(results, 10)
    print(model_ts, model_rn, model_ep)
    print(embeddings_cache_file)
    if lc:
        print('lowercase')
    print(model_file)
    print(fpath_model)
    print('gold:', gold_labels_dev)
    print('predicted:', predicted_labels_dev)

    assert isinstance(gold_labels_dev, list)
    assert isinstance(gold_labels_dev[0], int)
    assert len(gold_labels_dev) == len(predicted_labels_dev)

    # calculate scorer accuracy
    good_ids = set()
    wrong_ids = set()
    for g, p, instance_id in zip(gold_labels_dev, predicted_labels_dev, ids_dev):
        if g == p:
            good_ids.add(instance_id)
        else:
            wrong_ids.add(instance_id)
    acc_dev = len(good_ids) / (len(good_ids) + len(wrong_ids))
    print("acc_dev = %.3f" % acc_dev)

    pred_acc = accuracy_score(y_true=gold_labels_dev, y_pred=predicted_labels_dev)
    print("accuracy_score = %.3f" % pred_acc)
