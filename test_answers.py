# save the best model for this run
from pprint import pprint
from keras.preprocessing import sequence
from sklearn.metrics import accuracy_score
from classes import Data
from keras.models import load_model
from main import get_predicted_labels
from results_evaluator import load_results, tprint
from keras import __version__ as kv
from argument_parser import FILES
import numpy as np
import data_loader
import vocabulary_embeddings_extractor
import six.moves.cPickle as cPickle

OLD_MODEL = False


def predict(model, ids, warrant0, warrant1, label, reason, claim, debate):
    if OLD_MODEL:
        predicted_probabilities_dev = model.predict(
            {
                'sequence_layer_warrant0_input': warrant0,
                'sequence_layer_warrant1_input': warrant1,
                'sequence_layer_reason_input': reason,
                'sequence_layer_claim_input': claim,
                'sequence_layer_debate_input': debate,
            },
        )
    else:
        predicted_probabilities_dev = model.predict(
            {
                'sequence_layer_input_warrant0': warrant0,
                'sequence_layer_input_warrant1': warrant1,
                'sequence_layer_input_reason': reason,
                'sequence_layer_input_claim': claim,
                'sequence_layer_input_debate': debate,
            },
        )

    # print(predicted_probabilities_dev)
    # print(predicted_probabilities_test)

    if label is not None:
        y_true = label
        y_pred = get_predicted_labels(predicted_probabilities_dev)

        assert isinstance(y_true, list)
        assert isinstance(y_true[0], int)
        assert len(y_true) == len(y_pred)

        # calculate scorer accuracy
        good_ids = set()
        wrong_ids = set()
        for g, p, instance_id in zip(y_true, y_pred, ids):
            if g == p:
                good_ids.add(instance_id)
            else:
                wrong_ids.add(instance_id)
        acc = len(good_ids) / (len(good_ids) + len(wrong_ids))
        acc_score = accuracy_score(y_true=y_true, y_pred=y_pred)

        print('y_true:', y_true)
        print('y_pred:', y_pred)
        print("acc = %.3f" % acc)
        print("acc_score = %.3f" % acc_score)


def test_main():
    model_dir = 'out/'

    # model_file = 'model_2017-12-10_01-29-43-821365_rn01_ep15_ac0.712.hdf5'
    model_file = 'model_2018-01-11_02-30-22-312837_rn03_ep07_ac0.655.h5fs'

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

    # seeding is needed for deterministic OOV vector
    seed = int(model_info['pre_seed'])
    # print(seed, type(seed))
    np.random.seed(seed)

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

    np.set_printoptions(precision=6, threshold=50, edgeitems=None, linewidth=6000, suppress=True, nanstr=None,
                        infstr=None, formatter=None)

    if False:
        pprint(word_to_indices_map)
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
    (d.train_ids, d.train_warrant0, d.train_warrant1, d.train_label, d.train_reason, d.train_claim, d.train_debate) = \
        data_loader.load_single_file(FILES['train_swap'], word_to_indices_map, lc=lc)
    # dev
    print('loading dev data')
    (d.dev_ids, d.dev_warrant0, d.dev_warrant1, d.dev_label, d.dev_reason, d.dev_claim, d.dev_debate) = \
        data_loader.load_single_file(FILES['dev'], word_to_indices_map, lc=lc)
    # test
    print('loading test data')
    (d.test_ids, d.test_warrant0, d.test_warrant1, d.test_label, d.test_reason, d.test_claim, d.test_debate) = \
        data_loader.load_single_file(FILES['test'], word_to_indices_map, lc=lc, no_labels=True)

    # pad all sequences
    # train
    (d.train_warrant0, d.train_warrant1, d.train_reason, d.train_claim, d.train_debat) = [
        sequence.pad_sequences(x, maxlen=model_info['padding'].item()) for x in
        (d.train_warrant0, d.train_warrant1, d.train_reason, d.train_claim, d.train_debate)]
    # dev
    (d.dev_warrant0, d.dev_warrant1, d.dev_reason, d.dev_claim, d.dev_debate) = [
        sequence.pad_sequences(x, maxlen=model_info['padding'].item()) for x in
        (d.dev_warrant0, d.dev_warrant1, d.dev_reason, d.dev_claim, d.dev_debate)]
    # test
    (d.test_warrant0, d.test_warrant1, d.test_reason, d.test_claim, d.test_debate) = [
        sequence.pad_sequences(x, maxlen=model_info['padding'].item()) for x in
        (d.test_warrant0, d.test_warrant1, d.test_reason, d.test_claim, d.test_debate)]

    print('Keras version:', kv)
    tprint(results, 10)
    print(model_ts, model_rn, model_ep)
    print(embeddings_cache_file)
    if lc:
        print('lowercase')
    print(model_file)
    print(fpath_model)
    # print(d)

    print("predict DEV:")
    predict(model, d.dev_ids, d.dev_warrant0, d.dev_warrant1, d.dev_label, d.dev_reason, d.dev_claim, d.dev_debate)
    print("------------")
    print("predict TEST:")
    predict(model, d.test_ids, d.test_warrant0, d.test_warrant1, None, d.test_reason, d.test_claim, d.test_debate)


if __name__ == '__main__':
    test_main()
