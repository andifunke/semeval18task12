import gensim.models.word2vec as wv
import gensim.models.fasttext as ftx
import pandas as pd
import spacy
from time import time
from results_evaluator import tprint
from constants import FILES

SPACY = spacy.load('en')


def tokenize_cell(item, lc=False):
    doc = SPACY(item)
    return [str(token).lower() if lc else str(token) for token in doc]


def get_labels(file):
    return pd.read_csv(file, sep='\t', header=0, index_col=['#id'],
                       usecols=['#id', 'correctLabelW0orW1'])


def get_data(file, pad=False, lowercase=False):
    df = pd.read_csv(file, sep='\t', header=0, index_col=['#id'],
                     usecols=['#id', 'warrant0', 'warrant1', 'reason', 'claim', 'debateTitle', 'debateInfo'])

    print('tokenize data')
    df = df.applymap(lambda x: tokenize_cell(x, lowercase))
    tprint(df, 10)

    if pad:
        print('pad data')
        df['warrant0'] = df['warrant0'].apply(lambda x: x + ['.$.'] * (36 - len(x)))
        df['warrant1'] = df['warrant1'].apply(lambda x: x + ['.$.'] * (36 - len(x)))
        df['reason'] = df['reason'].apply(lambda x: x + ['.$.'] * (54 - len(x)))
        df['claim'] = df['claim'].apply(lambda x: x + ['.$.'] * (15 - len(x)))
        df['debateTitle'] = df['debateTitle'].apply(lambda x: x + ['.$.'] * (12 - len(x)))
        df['debateInfo'] = df['debateInfo'].apply(lambda x: x + ['.$.'] * (32 - len(x)))
    else:
        df = df.applymap(lambda x: x + ['.$.'])

    return df


def custom_embedding_builder_main(padding=False):
    """ training a word2vec model from train, dev and test data.
        Different dimensionality can be used as well as different preprocessing (lowercase or not)
        as well as the CBOW and the Skip-gram model.
        Change any other parameter to taste.
    """
    t0 = time()
    padding = False

    print('read data')
    dfs = []
    for file in FILES.values():
        df = get_data(file, pad=padding)
        dfs.append(df)
    df = pd.concat(dfs)

    tprint(df, 10)

    fdata = './data/tokenized_data_{}.pkl'.format('padded' if padding else 'not-padded')
    print('saving to', fdata)
    df.to_pickle(fdata)

    sentences = []
    for row in df.itertuples():
        for s in row[1:]:
            sentences.append(s)
    # pprint(sentences, width=1600)

    for size in [25, 50, 100, 300]:
        word_ngrams = False
        sg = True
        hs = False
        iter_ = 20
        lc = False
        femb = DATA_DIR + 'custom_embedding_{}_{}_{}_iter{:d}_{:d}{}{}.vec'.format('ftx' if word_ngrams else 'wv2',
                                                                                   'sg' if sg else 'cb',
                                                                                   'hs' if hs else 'ns',
                                                                                   iter_,
                                                                                   size,
                                                                                   '_pad' if padding else '',
                                                                                   '_lc' if lc else '',
                                                                                   )
        print("starting w2v modelling, size={:d} for {}".format(size, femb))
        if not word_ngrams:
            model = wv.Word2Vec(sentences=sentences, sg=sg, hs=hs, size=size, alpha=0.025, window=5, min_count=0,
                                sample=1e-3, seed=1, workers=4,
                                min_alpha=0.0001, negative=5, cbow_mean=1, iter=iter_, null_word=0,
                                sorted_vocab=1, batch_words=10000)
        else:
            model = ftx.FastText(sentences=sentences, sg=sg, hs=hs, size=size, alpha=0.025, window=5, min_count=0,
                                 word_ngrams=word_ngrams, sample=1e-3, seed=1, workers=4,
                                 min_alpha=0.0001, negative=5, cbow_mean=1, iter=iter_, null_word=0, min_n=3,
                                 max_n=6, sorted_vocab=1, bucket=2000000, batch_words=10000)
        print("saving vectors to {}".format(femb))
        model.save(femb)

    print("done in {:f}s".format(time() - t0))


if __name__ == '__main__':
    DATA_DIR = './embedding_caches/'
    # nltk.data.load('tokenizers/punkt/english.pickle')
    custom_embedding_builder_main()
