import gensim.models.word2vec as wv
import gensim.models.fasttext as ftx
import numpy as np
import pandas as pd
import spacy
from time import time
from results_evaluator import tprint


def tokenize(item, lc=False):
    doc = SPACY(item)

    # cheeting on bad sentence-split:
    if item.lower() == 'Are Casinos Too Much of a Gamble?'.lower():
        texts = [[str(word).lower() if lc else str(word) for word in doc]]
    else:
        texts = [[str(word).lower() if lc else str(word) for word in sent] for sent in doc.sents]

    return texts


def sentences_from_corpus(corpus, lowercase=False):
    """ returns a list of lists of strings.
    The outer list contains all sentences while the inner lists contain all tokens as string. """
    df = pd.DataFrame()

    print('\n>>> getting sentences from {}{}'.format(corpus, ' - lowercase' if lowercase else ''))
    sentences = []
    sentence = []
    last_sentence_id = 1
    for row in df.itertuples():
        sentence_id = row[2]
        token = row[4]
        if sentence_id > last_sentence_id:
            sentences.append(sentence)
            sentence = []
            last_sentence_id = sentence_id
        sentence.append(token.lower() if lowercase else token)
    sentences.append(sentence)
    return sentences


def custom_embedding_builder_main():
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
        tprint(df, 10)
        for tple in df.itertuples():
            for item in tple[1:]:
                sentences.extend(tokenize(item))
                sentences_lc.extend(tokenize(item, lc=True))

    for lowercase in [False, True]:
        lc = '_lc' if lowercase else ''
        s = sentences_lc if lowercase else sentences
        for sg in [1]:  # use [0] or [0, 1] for (additional) CBOW
            mdl = 'cb' if sg == 0 else 'sg'
            for size in [100, 300]:
                    print("starting w2v {} modelling, size={:d}".format(mdl, size))
                    # model = wv.Word2Vec(s, size=size, alpha=0.025, window=5, min_count=0, max_vocab_size=None,
                    #                     sample=0.001, seed=1, workers=4, min_alpha=0.0001, sg=sg, hs=0, negative=5,
                    #                     cbow_mean=1, iter=5, null_word=0, trim_rule=None, sorted_vocab=1,
                    #                     batch_words=10000, compute_loss=False)
                    model = ftx.FastText(s, size=size, alpha=0.025, window=5, min_count=0, max_vocab_size=None,
                                         sample=0.001, seed=1, workers=4, min_alpha=0.0001, sg=sg, hs=1, negative=5,
                                         cbow_mean=1, iter=20, null_word=0, trim_rule=None, sorted_vocab=1,
                                         batch_words=10000)
                    fname = DATA_DIR + 'custom_embedding_ftx_hs_iter20_{}_{:d}{}.vec'.format(mdl, size, lc)
                    print("saving vectors to {}".format(fname))
                    model.save(fname)

    print("done in {:f}s".format(time() - t0))


if __name__ == '__main__':
    DATA_DIR = './embedding_caches/'
    SPACY = spacy.load('en')
    # nltk.data.load('tokenizers/punkt/english.pickle')
    custom_embedding_builder_main()
