""" pre-tokenizes and case-folds the data set to prevent randomness """
import pandas as pd
import ast
from constants import *
import spacy

SPACY = spacy.load('en')
path = '../data/preprocessed/'


def get_data_strings(lc=True):
    if lc:
        return pd.read_table(path + 'preprocessed_lc.tsv')
    else:
        return pd.read_table(path + 'preprocessed.tsv')


def get_data_indexes(lc=True):
    if lc:
        return pd.read_table(path + 'preprocessed_index_lc.tsv')
    else:
        return pd.read_table(path + 'preprocessed_index.tsv')


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
    fname = path + 'preprocessed'
    fname = fname + '_lc' if lc else fname
    print('saving to', fname)
    df.to_csv(fname + '.tsv', sep='\t', index=False)

    # create an word to index mapping
    voc = pd.Series(vocabulary, name='frequency')
    voc.index.name = 'words'
    voc.sort_index(inplace=True)
    voc.sort_values(inplace=True, ascending=False, kind="mergesort")
    fname = path + 'words_to_freq'
    fname = fname + '_lc' if lc else fname
    print('saving to', fname)
    voc.to_csv(fname + '.tsv', sep='\t', index=True)

    df[CONTENT] = df[CONTENT].applymap(lambda lst: [voc.index.get_loc(token)+3 for token in lst])
    fname = path + 'preprocessed_index'
    fname = fname + '_lc' if lc else fname
    print('saving to', fname)
    df.to_csv(fname + '.tsv', sep='\t', index=False)


if __name__ == "__main__":
    preprocess()
