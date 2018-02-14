import json
import pandas as pd
from data_analyzer import tokenize_cell
from constants import *


def preprocess():
    print('read data')
    dfs = []
    for set_type, file in FILES.items():
        print(file)
        df = pd.read_csv(file, sep='\t', dtype={'correctLabelW0orW1': bool})
        df['set'] = set_type
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    df = df[KEYS]
    df[CONTENT] = df[CONTENT].applymap(lambda s: tokenize_cell(s, lc=True))

    fdata = './data/preprocessed'
    print('saving to', fdata)
    df.to_csv(fdata + '.tsv', sep='\t', index=False)

    with open('./embeddings/custom_embeddings_freq_lc.json', 'r') as fp:
        freq = json.load(fp)
    df[CONTENT] = df[CONTENT].applymap(lambda lst: [freq.get(token, token) for token in lst])
    fdata = './data/preprocessed_freq'
    print('saving to', fdata)
    df.to_csv(fdata + '.tsv', sep='\t', index=False)
    # df.to_pickle(fdata + '.pkl')


if __name__ == "__main__":
    preprocess()
