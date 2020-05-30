"""
Training custom word embeddings.
"""

import gensim.models.word2vec as wv
import pandas as pd
import spacy
from time import time


def tokenize(item, lc=False, corrections=False):
    doc = SPACY(item)
    # Correcting on bad sentence-split:
    if corrections and item.lower() == 'Are Casinos Too Much of a Gamble?'.lower():
        texts = [[str(word).lower() if lc else str(word) for word in doc]]
    else:
        texts = [[str(word).lower() if lc else str(word) for word in sent] for sent in doc.sents]
    return texts


def custom_embedding_builder_main(add_wikipedia=False):
    """
    Trains a word2vec model from train, dev and test data.

    Different dimensionality can be used as well as different pre-processing (lowercase or not)
    as well as the CBOW and the Skip-gram model. Change any other parameter to taste.
    """
    t0 = time()

    files = [
        './data/dev/dev-only-data.txt',
        './data/test/test-only-data.txt',
        './data/train/train-full.txt'
    ]

    sentences = []
    sentences_lc = []
    for file in files:
        df = pd.read_csv(
            file, sep='\t', header=0, index_col=['#id'], usecols=[
                '#id', 'warrant0', 'warrant1', 'reason', 'claim', 'debateTitle', 'debateInfo'
            ]
        )
        print(file)
        for tple in df.itertuples():
            for item in tple[1:]:
                sentences_lc.extend(tokenize(item, lc=True))

    if add_wikipedia:
        # add wikipedia corpus
        wiki_fname = './embedding_caches/wikipedia_corpus.txt'
        with open(wiki_fname, 'r') as f:
            wikipedia = f.readlines()
        print('tokenize wikipedia')
        for line in wikipedia:
            line = line.strip()
            if line == '' or line[0] == '=':
                continue
            sentences_lc.extend(tokenize(line, lc=True, corrections=False))

    del df
    del wikipedia

    for lowercase in [True]:
        lc = '_lc' if lowercase else ''
        s = sentences_lc if lowercase else sentences
        # TODO: Add option to arguments
        for sg in [1]:  # use [0] or [0, 1] for (additional) CBOW
            mdl = 'cb' if sg == 0 else 'sg'
            for size in [100, 300]:
                print("starting w2v {} modelling, size={:d}".format(mdl, size))
                model = wv.Word2Vec(
                    s, size=size, alpha=0.025, window=5, min_count=0, max_vocab_size=None,
                    sample=0.001, seed=1, workers=4, min_alpha=0.0001, sg=sg, hs=0, negative=5,
                    cbow_mean=1, iter=20, null_word=0, trim_rule=None, sorted_vocab=1,
                    batch_words=10000, compute_loss=False
                )
                # TODO: Add option to arguments
                # model = ftx.FastText(s, size=size, alpha=0.025, window=5, min_count=0, max_vocab_size=None,
                #                      sample=0.001, seed=1, workers=4, min_alpha=0.0001, sg=sg, hs=1, negative=5,
                #                      cbow_mean=1, iter=20, null_word=0, trim_rule=None, sorted_vocab=1,
                #                      batch_words=10000)
                w = '_wiki' if add_wikipedia else ''
                fname = DATA_DIR + 'custom_embedding_w2v_hs_iter20_{}_{:d}{}{}.vec'.format(mdl, size, lc, w)
                print("saving vectors to {}".format(fname))
                model.save(fname)

    print("done in {:f}s".format(time() - t0))


if __name__ == '__main__':
    DATA_DIR = './embedding_caches/'
    SPACY = spacy.load('en')
    custom_embedding_builder_main(add_wikipedia=True)
