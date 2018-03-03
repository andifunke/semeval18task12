""" pre-tokenizes and case-folds the data set to prevent randomness """
import os
import ast
import json

import numpy as np
np.random.seed(0)

import pandas as pd
from sklearn.model_selection import train_test_split
import spacy
from constants import FILES, WARRANT0, WARRANT1, LABEL, KEYS, CONTENT

PREPRO_PATH = os.path.dirname(os.path.abspath(__file__)) + '/../data/preprocessed/'


def load_embedding(options: dict, seed=0):
    # TODO: 5) make embedding cache from fastText for own tokenization

    embedding = options['embedding'].split('.')[0]
    embedding_path = options['emb_dir'] + options['embedding'] + '.json'
    lc = True if ('_lc' in embedding) else False
    print('load embedding ... lowercase:', lc)

    # the words_to_freq* files are mainly interesting because of the index position of the tokens
    # the index position results from two reproducible sorts:
    # - 1) alphabetically and thus uniquely sorted
    # - 2) by frequency, stable sort
    if lc:
        indices_path = options['data_dir'] + 'preprocessed/' + 'words_to_freq_lc.tsv'
    else:
        indices_path = options['data_dir'] + 'preprocessed/' + 'words_to_freq.tsv'
    word_indices_df = pd.read_csv(indices_path, '\t', names=['freq'], squeeze=False, index_col=0, header=None,
                                  skip_blank_lines=False)

    # loading the words to vectors mapping and converting the dict to a DataFrame
    with open(embedding_path, 'r') as fp:
        print(embedding_path)
        words_to_vectors = json.load(fp)

    words_to_vectors_df = pd.DataFrame.from_dict(words_to_vectors, orient='index')

    # determining the dimensionality of the embedding
    dimension = len(words_to_vectors['the'])
    # for padding we add a constant vector, for start of sequence and OOV we add random vectors
    np.random.seed(seed)
    vector_padding = np.asarray([0.0] * dimension)
    vector_start_of_sequence = 2 * 0.1 * np.random.rand(dimension) - 0.1
    vector_oov = 2 * 0.1 * np.random.rand(dimension) - 0.1
    # adding the special vectors to the DataFrame
    words_to_vectors_df.loc['$padding$'] = pd.Series(vector_padding)
    words_to_vectors_df.loc['$start_of_sequence$'] = pd.Series(vector_start_of_sequence)
    words_to_vectors_df.loc['$oov$'] = pd.Series(vector_oov)

    # combine (join) indices and vectors. By doing so, the vectors are placed at the correct position
    indices_to_vectors_df = word_indices_df.join(words_to_vectors_df, how='left')
    indices_to_vectors_df.drop('freq', axis=1, inplace=True)

    # fill missing with OOV-vector
    indices_to_vectors_df.fillna(words_to_vectors_df.loc['$oov$'], axis='index', inplace=True)

    assert len(word_indices_df) == len(indices_to_vectors_df)
    assert word_indices_df.index.equals(indices_to_vectors_df.index)

    options['vocabulary'] = len(words_to_vectors)
    options['dimension'] = dimension
    options['lowercase'] = lc

    return indices_to_vectors_df.values


def get_vectors(sequence, word_vectors, lowercase, zeros):
    return [
        word_vectors[token.lower() if lowercase else token]
        if token in word_vectors
        else zeros
        for token in sequence
    ]


def split_train_dev_test(df: pd.DataFrame, train_ratio=0.6145, dev_test_ratio=0.416, seed: int=0):
    if train_ratio is None:
        train_ratio = 0.6145
    if dev_test_ratio is None:
        dev_test_ratio = 0.416

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
    print(len(df_swapped))
    return df_swapped


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


def read_data(data_dir: str=None, indexed=False, lc: bool=True):
    if data_dir:
        path = data_dir + 'preprocessed/'
    else:
        path = PREPRO_PATH

    suffix1 = '_index' if indexed else ''
    suffix2 = '_lc' if lc else ''

    return pd.read_table(path + 'preprocessed{}{}.tsv'.format(suffix1, suffix2))


def load_data(dataset: [str, list]=None, data_dir: dict=None, lc: bool=True, indexed=False):
    """ get data """
    print('load data for', dataset, '... lowercase:', lc)
    if indexed:
        df = read_data(data_dir, indexed=True, lc=lc)
    else:
        df = read_data(data_dir, indexed=False, lc=lc)
    if isinstance(dataset, str) and dataset in FILES.keys():
        df = df[df['set'] == dataset]
    elif isinstance(dataset, list):
        df = df[df['set'].isin(dataset)]
    else:
        return None
    df['pred'] = False
    return df.reset_index(drop=True)


def get_train_dev_test(options: dict):
    data_dir = options['data_dir'] if 'data_dir' in options else None
    lc = options['lowercase'] if 'lowercase' in options else True
    dev_test_ratio = options['dev_test_ratio'] if 'dev_test_ratio' in options else None
    padding_size = options['padding'] if 'padding' in options else 54

    if 'alt_split' in options and options['alt_split']:
        df = load_data(dataset=['train', 'dev', 'test'], data_dir=data_dir, indexed=True, lc=lc)
        print('pad sequences')
        df[CONTENT] = df[CONTENT].applymap(lambda sequence: pad(sequence, padding_size=padding_size))
        df_train, df_dev, df_test = split_train_dev_test(df, dev_test_ratio=dev_test_ratio)
        df_train = add_swap(df_train)
    else:
        df = load_data(dataset=['train_swap', 'dev', 'test'], data_dir=data_dir, indexed=True, lc=lc)
        print('pad sequences')
        df[CONTENT] = df[CONTENT].applymap(lambda sequence: pad(sequence, padding_size=padding_size))
        print('using default split and swap')
        df_train = df[df['set'] == 'train_swap'].copy(deep=True)
        df_dev = df[df['set'] == 'dev'].copy(deep=True)
        df_test = df[df['set'] == 'test'].copy(deep=True)

    return df_train, df_dev, df_test


def tokenize_cell(spacy_instance, item: str, vocabulary: dict=None, lc: bool=True):
    doc = spacy_instance(item)
    tokens = [str(token).lower() if lc else str(token) for token in doc]
    if vocabulary is not None:
        for token in tokens:
            vocabulary[token] = vocabulary.get(token, 0) + 1
    return tokens


def preprocess(lc=True):
    spacy_instance = spacy.load('en')

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
    df[CONTENT] = df[CONTENT].applymap(lambda s: tokenize_cell(spacy_instance, s, vocabulary, lc=lc))
    fname = PREPRO_PATH + 'preprocessed'
    fname = fname + '_lc' if lc else fname
    print('saving to', fname)
    df.to_csv(fname + '.tsv', sep='\t', index=False)

    # create an word to index mapping
    # TODO: add offset
    # TODO: check if '.' is converted to '0'
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
