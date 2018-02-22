""" parsing arguments and setting default values """
import os
import argparse
from constants import *


def get_options():
    emb_files_list = list(EMB_FILES.keys()) + list(EMB_FILES.values())

    parser = argparse.ArgumentParser(description='semeval 18 task 12 - training project')

    parser.add_argument('--verbose', default=1, type=int,
                        choices=[0, 1],
                        help='project verbosity should be set to 0 for deployment on cluster')
    parser.add_argument('--classifier', default='LSTM_01', type=str)
    parser.add_argument('--lstm_size', default=64, type=int,
                        help='size of the lstm hidden layer')
    parser.add_argument('--dense_factor', default=1.0, type=float,
                        help='factor of the additional dense hidden layer')
    parser.add_argument('--dropout', default=0.9, type=float,
                        help='dropout must be float(x) with 0 < x <= 1. unchecked')
    parser.add_argument('--epochs', default=20, type=int,
                        help='maximum number of epochs. training may terminate eralier.')
    parser.add_argument('--patience', default=5, type=int,
                        help='for early stopping.')
    parser.add_argument('--padding', default=100, type=int,
                        help='length of padded vectors. choose <= 0 for maximum padding length (no truncation).')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='size of batch')
    parser.add_argument('--pre_seed', default=12345, type=int,
                        help='add an offset to the seed. The actual seed per run is pre_seed + run')
    parser.add_argument('--run', default=1, type=int,
                        help='chose the first run index (1-based indexing')
    parser.add_argument('--runs', default=3, type=int,
                        help='number of runs (starting with index of run)')
    parser.add_argument('--embedding', default='w2v', type=str,
                        choices=emb_files_list, help='specify first embedding')
    parser.add_argument('--embedding2', default='', type=str,
                        choices=emb_files_list, help='a second embedding is used when specified')
    parser.add_argument('--optimizer', default='adam', type=str,
                        choices=['sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam', 'adamax', 'nadam', 'tfoptimizer'],
                        help='specify optimizer function')
    parser.add_argument('--loss', default='binary_crossentropy', type=str,
                        choices=['binary_crossentropy', 'mean_squared_error', 'mean_absolute_error',
                                 'mean_absolute_percentage_error', 'mean_squared_logarithmic_error',
                                 'squared_hinge', 'hinge', 'categorical_hinge', 'logcosh',
                                 'categorical_crossentropy', 'sparse_categorical_crossentropy',
                                 'kullback_leibler_divergence', 'poisson, cosine_proximity'],
                        help='specify loss function')
    activations = ['relu', 'softmax', 'elu', 'selu', 'softplus', 'softsign', 'tanh', 'sigmoid',
                   'sigmoid', 'linear', 'leakyrelu', 'prelu', 'elu', 'thresholdedrelu']
    parser.add_argument('--activation1', default='relu', type=str,
                        choices=activations, help='specify activation function for inner layers')
    parser.add_argument('--activation2', default='sigmoid', type=str,
                        choices=activations, help='sepcify activation function for output layer')
    parser.add_argument('--vsplit', default=0.1, type=float,
                        help='fraction used for cross validation. should be float(x) with 0 < x < 1. unchecked')
    parser.add_argument('--rich', default=3, type=int,
                        help='need to look it up myself :)')
    parser.add_argument('--code_path', default=os.getcwd()+'/', type=str,
                        help='specify code path on cluster. default is project dir.')
    parser.add_argument('--save_path', default=os.getcwd()+'/', type=str,
                        help='specify save path on cluster. default is project dir.')
    parser.add_argument('--emb_dir', default='../embeddings/', type=str,
                        help='usually leave as is ')
    parser.add_argument('--out_path', default='out/', type=str,
                        help='usually leave as is')
    parser.add_argument('--save_models', default=False, type=bool,
                        help='save model with each improving epoch')
    parser.add_argument('--true_lc', default=False, type=bool,
                        help='use lowercase when embedding was trained on lowercase tokens')
    parser.add_argument('--threshold', default=0.67, type=float,
                        help='saving data only for models exceeding this threshold')
    parser.add_argument('--spacy', default='en', type=str,
                        help='spacy shortcut or path to language model')
    parser.add_argument('--system', default='local', type=str,
                        choices={'local', 'hpc'},
                        help='shortcut for several parameters: emb_dir, spacy, runs, epochs, threshold')
    parser.add_argument('--comment', default='', type=str)

    options = vars(parser.parse_args())
    if options['padding'] < 1:
        options['padding'] = None
    if options['system'] == 'hpc':
        options['emb_dir'] = '../embeddings/'
        options['spacy'] = '~/.local/lib/python3.4/site-packages/en_core_web_sm/en_core_web_sm-1.2.0/'
        options['runs'] = 10
        options['epochs'] = 20
        options['threshold'] = 0.68
    if options['embedding'] in EMB_FILES.keys():
        options['embedding'] = EMB_FILES[options['embedding']]
    if options['embedding2'] in EMB_FILES.keys():
        options['embedding2'] = EMB_FILES[options['embedding2']]

    return options
