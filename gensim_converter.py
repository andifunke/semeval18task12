import json
import re

from os import listdir
from pprint import pprint

import gensim.models.word2vec as wv
import six.moves.cPickle as cPickle


def gensim_wrapper_main():
    """
    since gensim cannot be installed on the hpc this small script replaces
    gensim KeyedVectors by plain old vanilla python dicts
    """

    files = [f for f in listdir(DATA_DIR) if re.match('^custom_embedding_.*\.vec$', f)]

    for fname in files:
        print('loading model', fname)
        model = wv.Word2Vec.load(DATA_DIR + fname)
        print('building dict')

        # json
        vector_dict = {word: [float(x) for x in model.wv[word]] for word in model.wv.vocab}
        print('saving dict to json')
        with open(DATA_DIR + fname + '.json', 'w', encoding='utf8') as f:
            json.dump(vector_dict, f, ensure_ascii=False)

        # pickle
        vector_dict = {word: model.wv[word] for word in model.wv.vocab}
        print('saving dict to pickle')
        with open(DATA_DIR + fname + '.pickle', 'wb') as f:
            cPickle.dump(vector_dict, f)


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
        model = wv.Word2Vec.load(DATA_DIR + fname)
        print('building dict')

        # json
        vector_dict = {word: [float(x) for x in model.wv[word]] for word in vocabulary}
        print('saving dict to json')
        with open(DATA_DIR + fname + '.json', 'w', encoding='utf8') as f:
            json.dump(vector_dict, f, ensure_ascii=False)

        # pickle
        vector_dict = {word: model.wv[word] for word in vocabulary}
        print('saving dict to pickle')
        with open(DATA_DIR + fname + '.pickle', 'wb') as f:
            cPickle.dump(vector_dict, f)


if __name__ == '__main__':
    DATA_DIR = './embedding_caches/new/'
    # gensim_wrapper_main()
    gensim_wikipedia_embedding_wrapper()
    # quit()

    verify = True

    if verify:
        fname = 'custom_embedding_w2v_hs_iter20_sg_300_lc_wiki.vec'
        with open(DATA_DIR + fname + '.json', 'r') as f:
            vector_dict = json.load(f)

        print(vector_dict.keys())
        print(len(vector_dict))

        import numpy as np

        with open(DATA_DIR + fname + '.pickle', 'rb') as f:
            vector_dict = cPickle.load(f)

        np.set_printoptions(precision=6, threshold=50, edgeitems=3, linewidth=1000, suppress=True, nanstr=None,
                            infstr=None, formatter=None)

        pprint(vector_dict, width=1000)
        print(len(vector_dict))
