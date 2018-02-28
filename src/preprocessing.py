""" pre-tokenizes and case-folds the data set to prevent randomness """
import os
import ast
import json

import numpy as np

from results_evaluator import tprint

np.random.seed(0)

from theano.scalar import float32
import pandas as pd
from gensim.models import word2vec as wv
from sklearn.model_selection import train_test_split
import spacy
from constants import FILES, WARRANT0, WARRANT1, LABEL, KEYS, EMB_DIR, CONTENT, CONTENT_MIN

SPACY = spacy.load('en')
PREPRO_PATH = os.path.dirname(os.path.abspath(__file__)) + '/' + '../data/preprocessed/'


def get_embedding(options: dict, seed=0):
    # TODO: 5) make embedding cache from fastText for own tokenization

    print('load embedding')

    # TODO: 4) check, if <code_path> works on HPC
    code_path = os.path.dirname(os.path.abspath(__file__)) + '/'
    embedding = options['embedding'].split('.')[0]
    print('embedding:', embedding)
    embedding_path = code_path + options['emb_dir'] + options['embedding'] + '.json'
    lc = True if ('_lc' in embedding) else False
    print('get_embeddings ... lowercase:', lc)

    if lc:
        freq_path = PREPRO_PATH + 'words_to_freq_lc.tsv'
    else:
        freq_path = PREPRO_PATH + 'words_to_freq.tsv'

    with open(embedding_path, 'r') as fp:
        print(embedding_path)
        words_to_vectors = json.load(fp)
        print('len(words_to_vectors)', len(words_to_vectors))

    word_indices_df = pd.read_csv(freq_path, '\t', names=['freq'], squeeze=False, index_col=0, header=None,
                                  skip_blank_lines=False)
    print('len(words_to_indices)', len(word_indices_df))
    print(word_indices_df)

    for word, vector in words_to_vectors.items():
        # TODO: 1) add vectors as MultiIndex
        word_indices_df.loc[word, 'vec'] = word

    # for start of sequence and OOV we add random vectors
    dimension = len(words_to_vectors['the'])
    print(dimension)

    np.random.seed(seed)
    vector_padding = np.asarray([0.0] * dimension)
    vector_start_of_sequence = 2 * 0.1 * np.random.rand(dimension) - 0.1
    vector_oov = 2 * 0.1 * np.random.rand(dimension) - 0.1
    # and add them to the embeddings map (as the first three values)
    # TODO: 2) add vectors as MultiIndex
    word_indices_df.loc['$padding$', 'vec'] = 'abc'
    word_indices_df.loc['$start_of_sequence$', 'vec'] = 'def'
    word_indices_df.loc['$oov$', 'vec'] = 'ghi'

    tprint(word_indices_df)
    print('length:', len(word_indices_df))
    quit()

    return  # TODO: 3) return 2D ndarray


def get_vectors(sequence, word_vectors, lowercase, zeros):
    return [
        word_vectors[token.lower() if lowercase else token]
        if token in word_vectors
        else zeros
        for token in sequence
    ]


def get_xy(df: pd.DataFrame, options, padding_size=54):
    print('get x_y')

    lowercase = options['lowercase']
    print('lowercase', lowercase)
    fname = EMB_DIR + options['wv_file'] + '.vec'
    print('loading embeddings from', fname)
    model = wv.Word2Vec.load(fname)
    word_vectors = model.wv
    options['dims'] = dims = len(word_vectors['a'])
    print('embedding dimensions', dims)
    zeros = np.zeros(dims)

    # pad data
    df[CONTENT] = df[CONTENT].applymap(lambda sequence:
                                       pad(sequence, padding_size=padding_size, padding_symbol='.$.'))
    # replace with word vectors
    df[CONTENT] = df[CONTENT].applymap(lambda sequence: get_vectors(sequence, word_vectors, lowercase, zeros))

    x_list = df[CONTENT_MIN].values.tolist()
    y_list = df[LABEL].values.tolist()
    x = np.asarray(x_list)
    x = np.reshape(x, (x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]), order='C')
    y = np.asarray(y_list, dtype=bool, order='C')
    assert len(x) == len(y)
    return x, y


def get_data_strings(lc=True):
    if lc:
        return pd.read_table(PREPRO_PATH + 'preprocessed_lc.tsv')
    else:
        return pd.read_table(PREPRO_PATH + 'preprocessed.tsv')


def get_data_indexes(lc=True):
    if lc:
        return pd.read_table(PREPRO_PATH + 'preprocessed_index_lc.tsv')
    else:
        return pd.read_table(PREPRO_PATH + 'preprocessed_index.tsv')


def get_data(dataset: [str, list]=None, lowercase: bool=True, use_indexes=False):
    """ get data """
    print('loading data for', dataset)
    if use_indexes:
        df = get_data_indexes(lc=lowercase)
    else:
        df = get_data_strings(lc=lowercase)
    if isinstance(dataset, str) and dataset in FILES.keys():
        df = df[df['set'] == dataset]
    elif isinstance(dataset, list):
        df = df[df['set'].isin(dataset)]
    else:
        return None
    return df.reset_index(drop=True)


def split_train_dev_test(df: pd.DataFrame, train_ratio=0.6145, dev_test_ratio=0.416, seed: int=0):
    print('split train-dev-test')
    np.random.seed(seed)
    train, dev_test = train_test_split(df, test_size=None, train_size=train_ratio)
    dev, test = train_test_split(dev_test, test_size=None, train_size=dev_test_ratio)
    return train, dev, test


def add_swap(df: pd.DataFrame):
    print('add swap')
    # upsample the train set with swapped warrant 0<->1
    df_copy = df.copy(deep=True)
    df_copy = df_copy.rename(columns={WARRANT0: WARRANT1, WARRANT1: WARRANT0})
    df_copy[LABEL] ^= 1
    df_swapped = pd.concat([df, df_copy]).sort_index(kind="mergesort").reset_index(drop=True)
    df_swapped = df_swapped[KEYS]
    return df_swapped


def tokenize_cell(item, vocabulary=None, lc=True):
    doc = SPACY(item)
    tokens = [str(token).lower() if lc else str(token) for token in doc]
    if vocabulary is not None:
        for token in tokens:
            vocabulary[token] = vocabulary.get(token, 0) + 1
    return tokens


def pad(sequence: list, padding_size: int, padding_symbol=0):
    sequence = ast.literal_eval(sequence)
    length = len(sequence)
    if length > padding_size:
        s = sequence[-padding_size:]
    elif length < padding_size:
        s = ([padding_symbol] * (padding_size - length)) + sequence
    else:
        s = sequence
    assert len(s) == padding_size
    return s


def preprocess(lc=True):
    print('read data')
    dfs = []
    for subset in FILES.keys():
        file = FILES[subset]
        print(file)
        df = pd.read_csv(file, sep='\t')
        df['set'] = subset
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    df = df[KEYS]
    vocabulary = dict()
    df[CONTENT] = df[CONTENT].applymap(lambda s: tokenize_cell(s, vocabulary, lc=lc))
    fname = PREPRO_PATH + 'preprocessed'
    fname = fname + '_lc' if lc else fname
    print('saving to', fname)
    df.to_csv(fname + '.tsv', sep='\t', index=False)

    # create an word to index mapping
    # TODO: add offset
    voc = pd.Series(vocabulary, name='frequency')
    voc.index.name = 'words'
    voc.sort_index(inplace=True)
    voc.sort_values(inplace=True, ascending=False, kind="mergesort")
    fname = PREPRO_PATH + 'words_to_freq'
    fname = fname + '_lc' if lc else fname
    print('saving to', fname)
    voc.to_csv(fname + '.tsv', sep='\t', index=True)

    df[CONTENT] = df[CONTENT].applymap(lambda lst: [voc.index.get_loc(token)+3 for token in lst])
    fname = PREPRO_PATH + 'preprocessed_index'
    fname = fname + '_lc' if lc else fname
    print('saving to', fname)
    df.to_csv(fname + '.tsv', sep='\t', index=False)


if __name__ == "__main__":
    preprocess()
