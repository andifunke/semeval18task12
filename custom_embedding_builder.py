import gensim.models.word2vec as wv
import gensim.models.fasttext as ftx
import numpy as np
import pandas as pd
import spacy
from time import time
from results_evaluator import tprint


def tokenize(item, lc=False, cheet=False):
    doc = SPACY(item)
    # cheeting on bad sentence-split:
    if cheet and item.lower() == 'Are Casinos Too Much of a Gamble?'.lower():
        texts = [[str(word).lower() if lc else str(word) for word in doc]]
    else:
        texts = [[str(word).lower() if lc else str(word) for word in sent] for sent in doc.sents]
    # print(texts)
    return texts


def custom_embedding_builder_main(add_wikipedia=False):
    """ training a word2vec model from train, dev and test data.
        Different dimensionality can be used as well as different preprocessing (lowercase or not)
        as well as the CBOW and the Skip-gram model.
        Change any other parameter to taste.
    """
    t0 = time()

    files = ['./data/dev/dev-only-data.txt', './data/test/test-only-data.txt', './data/train/train-full.txt']

    sentences = []
    sentences_lc = []
    for file in files:
        df = pd.read_csv(file, sep='\t', header=0, index_col=['#id'],
                         usecols=['#id', 'warrant0', 'warrant1', 'reason', 'claim', 'debateTitle', 'debateInfo'])
        print(file)
        # tprint(df, 10)
        for tple in df.itertuples():
            for item in tple[1:]:
                # sentences.extend(tokenize(item))
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
            # sentences.extend(tokenize(line, cheet=False)
            sentences_lc.extend(tokenize(line, lc=True, cheet=False))

    del df
    del wikipedia

    for lowercase in [True]:
        lc = '_lc' if lowercase else ''
        s = sentences_lc if lowercase else sentences
        for sg in [1]:  # use [0] or [0, 1] for (additional) CBOW
            mdl = 'cb' if sg == 0 else 'sg'
            for size in [300]:
                    print("starting w2v {} modelling, size={:d}".format(mdl, size))
                    model = wv.Word2Vec(s, size=size, alpha=0.025, window=5, min_count=0, max_vocab_size=None,
                                        sample=0.001, seed=1, workers=4, min_alpha=0.0001, sg=sg, hs=0, negative=5,
                                        cbow_mean=1, iter=5, null_word=0, trim_rule=None, sorted_vocab=1,
                                        batch_words=10000, compute_loss=False)
                    # model = ftx.FastText(s, size=size, alpha=0.025, window=5, min_count=0, max_vocab_size=None,
                    #                      sample=0.001, seed=1, workers=4, min_alpha=0.0001, sg=sg, hs=1, negative=5,
                    #                      cbow_mean=1, iter=20, null_word=0, trim_rule=None, sorted_vocab=1,
                    #                      batch_words=10000)
                    w = '_wiki' if add_wikipedia else ''
                    fname = DATA_DIR + 'custom_embedding_w2v_hs_iter05_{}_{:d}{}{}.vec'.format(mdl, size, lc, w)
                    print("saving vectors to {}".format(fname))
                    model.save(fname)

    print("done in {:f}s".format(time() - t0))


if __name__ == '__main__':
    DATA_DIR = './embedding_caches/'
    SPACY = spacy.load('en')
    # nltk.data.load('tokenizers/punkt/english.pickle')
    custom_embedding_builder_main(add_wikipedia=True)
