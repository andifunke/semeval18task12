""" converts gensim embeddings to json/pickle objetcs """
import json
import re
from os import listdir
from pprint import pprint
from gensim import models
import six.moves.cPickle as cPickle
import numpy as np
import pandas as pd


def embedding_cache_builder(fpath, lc=True):
    """
    create a json-cache file from a given .vec file.
    """
    print('loading model', fpath)
    model = models.KeyedVectors.load_word2vec_format(fpath)

    print('building dict')
    if lc:
        wtf = '../data/preprocessed/words_to_freq_lc.tsv'
    else:
        wtf = '../data/preprocessed/words_to_freq.tsv'
    vocabulary = pd.read_csv(wtf, sep='\t', header=None)
    vector_dict = dict()
    for token in vocabulary[0]:
        if token in model.wv:
            vector_dict[token] = model.wv[token].tolist()

    print(vector_dict)
    print(len(vector_dict))

    print('saving dict to json')
    x = '_lc' if lc else ''
    with open(fpath + x + '_cache.json', 'w', encoding='utf8') as fp:
        json.dump(vector_dict, fp, ensure_ascii=False, indent=1)


def gensim_wrapper_main():
    """
    since gensim cannot be installed on the hpc this small script replaces
    gensim KeyedVectors by plain old vanilla python dicts
    """

    files = [f for f in listdir(DATA_DIR) if re.match('^custom_embedding_.*\.vec$', f)]

    for fname in files:
        print('loading model', fname)
        model = models.Word2Vec.load(DATA_DIR + fname)
        print('building dict')

        # json
        vector_dict = {word: [float(x) for x in model.wv[word]] for word in model.wv.vocab}
        print('saving dict to json')
        with open(DATA_DIR + fname + '.json', 'w', encoding='utf8') as fp:
            json.dump(vector_dict, fp, ensure_ascii=False)

        # pickle
        vector_dict = {word: model.wv[word] for word in model.wv.vocab}
        print('saving dict to pickle')
        with open(DATA_DIR + fname + '.pickle', 'wb') as fp:
            cPickle.dump(vector_dict, fp)


def gensim_wikipedia_embedding_wrapper():
    """
    since gensim cannot be installed on the hpc this small script replaces
    gensim KeyedVectors by plain old vanilla python dicts
    for the wikipedia embeddings it also reduces the vector set to tokens in the vocabulary
    (i.e. building a cache file)
    """

    files = [f for f in listdir(DATA_DIR) if re.match('^custom_embedding_.*wiki\.vec$', f)]

    for fname in files:
        if '_lc' in fname:
            with open(DATA_DIR + 'custom_embeddings_freq_lc.json', 'r') as fp:
                vocabulary = json.load(fp)
        else:
            with open(DATA_DIR + 'custom_embeddings_freq.json', 'r') as fp:
                vocabulary = json.load(fp)

        print('loading model', fname)
        model = models.Word2Vec.Word2Vec.load(DATA_DIR + fname)
        print('building dict')

        # json
        vector_dict = {word: [float(x) for x in model.wv[word]] for word in vocabulary}
        print('saving dict to json')
        with open(DATA_DIR + fname + '.json', 'w', encoding='utf8') as fp:
            json.dump(vector_dict, fp, ensure_ascii=False)

        # pickle
        vector_dict = {word: model.wv[word] for word in vocabulary}
        print('saving dict to pickle')
        with open(DATA_DIR + fname + '.pickle', 'wb') as fp:
            cPickle.dump(vector_dict, fp)


if __name__ == '__main__':
    DATA_DIR = './embeddings/'
    #gensim_wikipedia_embedding_wrapper()
    embedding_cache_builder("/media/andreas/Linux_Data/old_workspace/sem_train/remote_vectors/fastText/wiki.en.vec")

    verify = False
    if verify:
        fname = 'custom_embedding_w2v_hs_iter20_sg_300_lc_wiki.vec'
        with open(DATA_DIR + fname + '.json', 'r') as f:
            vec_dict = json.load(f)
        print(vec_dict.keys())
        print(len(vec_dict))
        with open(DATA_DIR + fname + '.pickle', 'rb') as f:
            vec_dict = cPickle.load(f)
        np.set_printoptions(precision=6, threshold=50, edgeitems=3, linewidth=1000, suppress=True, nanstr=None,
                            infstr=None, formatter=None)
        pprint(vec_dict, width=1000)
        print(len(vec_dict))
