from pprint import pprint
import json
import pandas as pd
import spacy
from time import time
from collections import defaultdict


def vocabulary_analyzer_main():
    t0 = time()
    files = ['./data/dev/dev-only-data.txt', './data/test/test-only-data.txt', './data/train/train-full.txt']
    for file in files:
        df = pd.read_csv(file, sep='\t', header=0, index_col=['#id'],
                         usecols=['#id', 'warrant0', 'warrant1', 'reason', 'claim', 'debateTitle', 'debateInfo'])
        for tple in df.itertuples():
            for item in tple[1:]:
                doc = SPACY(item)
                for word in doc:
                    WORDS[str(word)] += 1
                    WORDS_LC[str(word).lower()] += 1
    pprint(WORDS)
    pprint(WORDS_LC)
    with open('custom_embeddings_freq.json', 'w') as fp:
        json.dump(WORDS, fp)
    with open('custom_embeddings_freq_lc.json', 'w') as fp:
        json.dump(WORDS_LC, fp)
    df_words = pd.DataFrame(sorted(WORDS.items()))
    df_words_lc = pd.DataFrame(sorted(WORDS_LC.items()))
    df_words.to_csv('custom_embeddings_freq.csv', sep='\t', index_label=False, index=False)
    df_words_lc.to_csv('custom_embeddings_freq_lc.csv', sep='\t', index_label=False, index=False)

    print("done in {:f}s".format(time() - t0))


if __name__ == '__main__':
    WORDS = defaultdict(int)
    WORDS_LC = defaultdict(int)
    SPACY = spacy.load('en')
    vocabulary_analyzer_main()
