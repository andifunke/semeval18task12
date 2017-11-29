import os
import argparse


# # argument parsing and setting default values
def get_options():
    emb_files = dict(
        w2v="embeddings_cache_file_word2vec.pkl.bz2",
        d2v="embeddings_cache_file_dict2vec.pkl.bz2",
        d2v_pf="embeddings_cache_file_dict2vec_prov_freq.pkl.bz2",
        d2v_pf_lc="embeddings_cache_file_dict2vec_prov_freq_lc.pkl.bz2",
        d2v_pf_lc2="embeddings_cache_file_dict2vec_prov_freq_lc2.pkl.bz2",
        ftx="embeddings_cache_file_fastText.pkl.bz2",
        ftx_pf="embeddings_cache_file_fastText_prov_freq.pkl.bz2",
        ftx_pf_lc="embeddings_cache_file_fastText_prov_freq_lc.pkl.bz2",
        ftx_pf_lc2="embeddings_cache_file_fastText_prov_freq_lc2.pkl.bz2",
    )

    parser = argparse.ArgumentParser(description='semeval 18 task 12 - training project')

    parser.add_argument('--verbose', default=1, type=int,
                        choices=[0, 1],
                        help='project verbosity should be set to 0 for deployment on cluster')
    parser.add_argument('--lstm_size', default=64, type=int,
                        help='size of the lstm hidden layer')
    parser.add_argument('--dropout', default=0.9, type=float,
                        help='dropout should be float(x) with 0 < x < 1. unchecked')
    parser.add_argument('--epochs', default=20, type=int,
                        help='maximum number of epochs. training may terminate eralier.')
    parser.add_argument('--padding', default=100, type=int,
                        help='length of padded vectors. choose -1 for maximum padding.')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='size of batch')
    parser.add_argument('--pre_seed', default=12345, type=int,
                        help='add an offset to the seed. The actual seed per run is pre_seed + run')
    parser.add_argument('--run', default=1, type=int,
                        help='chose the first run index (1-based indexing')
    parser.add_argument('--runs', default=3, type=int,
                        help='number of runs (starting with index of run)')
    parser.add_argument('--embedding', default='w2v', type=str,
                        choices=emb_files.keys(),
                        help='specify first embedding')
    parser.add_argument('--embedding2', default='', type=str,
                        choices=emb_files.keys(),
                        help='a second embedding is used when specified')
    parser.add_argument('--optimizer', default='adam', type=str,
                        help='specify optimizer function')
    parser.add_argument('--loss', default='binary_crossentropy', type=str,
                        help='specify loss function')
    parser.add_argument('--activation1', default='relu', type=str,
                        help='specify activation function for inner layers')
    parser.add_argument('--activation2', default='sigmoid', type=str,
                        help='sepcify activation function for output layer')
    parser.add_argument('--vsplit', default=0.1, type=float,
                        help='fraction used for cross validation. should be float(x) with 0 < x < 1. unchecked')
    parser.add_argument('--rich', default=3, type=int,
                        help='need to look it up myself :)')
    parser.add_argument('--code_path', default=os.getcwd()+'/', type=str,
                        help='specify code path on cluster. default is project dir.')
    parser.add_argument('--save_path', default=os.getcwd()+'/', type=str,
                        help='specify save path on cluster. default is project dir.')
    parser.add_argument('--emb_dir', default='embedding_caches/', type=str,
                        help='usually leave as is ')
    parser.add_argument('--out_path', default='out/', type=str,
                        help='usually leave as is')

    options = vars(parser.parse_args())
    print(options)

    # TODO:
    # max padding length = None
    # validation_data = dev (or dedicated batch)
    # shuffle = False
    # model
    # train with or without swap
    # full embeddings
    # combined embeddings (2x no_of_channels or 2x dimensionality)
    # reduced number of channels
    # spelling / pre-processing
    # backend -> re-write

    return options, emb_files
