import os
import pandas as pd


def extract_words_from_embedding(infile, outfile, engine):
    if engine == 'pandas':
        # faster, but uses more memory
        df = pd.read_csv(infile, usecols=[0, 1], sep=' ', header=None, engine='c', dtype={0: str, 1: str})
        df.sort_values(by=[0], inplace=True)
        df.to_csv(outfile, sep=' ', columns=0)
    else:
        # file too big => different approach
        fr = open(infile, 'r')
        out = []
        for line in fr:
            word = line.split()[0]
            out.append(word)
        out.sort()
        with open(outfile, 'w') as fw:
            fw.write(str(out))


if __name__ == "__main__":

    current_dir = os.getcwd()
    wv_folder = r'/home/andreas/workspace/sem_train/remote_vectors/'
    d2v = wv_folder + r'dict2vec/dict2vec-vectors-dim300.vec'
    ftx = wv_folder + r'fastText/wiki.en.vec'
    if True:
        vx.prepare_word_embeddings_cache([current_dir + '/data/'],
                                         current_dir + '/embeddings_cache_file_dict2vec_prov_freq_lc',
                                         embeddings_file_name=d2v,
                                         use_lower=True,
                                         use_provided_frequencies=True)
    if True:
        vx.prepare_word_embeddings_cache([current_dir + '/data/'],
                                         current_dir + '/embeddings_cache_file_fastText_prov_freq_lc',
                                         embeddings_file_name=ftx,
                                         use_lower=True,
                                         use_provided_frequencies=True)
    if False:
        extract_words_from_embedding(d2v, 'd2v_words.txt', engine='pandas')
        extract_words_from_embedding(ftx, 'ftx_words.txt')
