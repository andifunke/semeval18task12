""" combining predictions based on votes by a set of answer files """
import pandas as pd
import numpy as np
from scipy.stats import describe
from sklearn.metrics import accuracy_score
from os import listdir
import re
import matplotlib.pyplot as plt

from constants import LABEL, ID
from results_evaluator import tprint


np.set_printoptions(linewidth=8000)


def get_conf(prob):
    return np.abs(prob - 0.5)


def combine(y_true, y_pred, conf):
    """ confidence vote """
    conf_argmax = np.argmax(conf, axis=0)
    conf_vote = y_pred.T[np.arange(len(y_pred.T)), conf_argmax]
    acc_conf = accuracy_score(y_true=y_true, y_pred=conf_vote)
    # print('combined accuracy by confidence: {:.3f}'.format(acc_conf))

    """ majority vote """
    pred = np.mean(y_pred, axis=0)
    # in case of a tie use the predictions from the condifdence vote
    tie = np.isclose(pred, 0.5)
    pred[tie] = conf_vote[tie]
    pred = (pred >= 0.5)
    acc_major = accuracy_score(y_true=y_true, y_pred=pred)
    # print('combined accuracy by majority vote: {:.3f}'.format(acc_major))

    return acc_conf, acc_major


def plot(df, name, show=True):
    fig, ax = plt.subplots()
    df.plot(ax=ax)
    ax.yaxis.grid()
    ax.set_ylim(.5, .75)
    ax.set_xlabel('number of models')
    ax.set_ylabel('accuracy')
    ax.set_title('Ensemble scores')
    if show:
        plt.show()
    # fig.savefig('./figures/' + name + '.pdf')


def main():
    names = {
        # 'earlier_models': '/media/andreas/Linux_Data/hpc-semeval/out/',
        # 'tensorL05add': '/media/andreas/Linux_Data/hpc-semeval/tensorL05add/out/',
        # 'tensorL05add2': '/media/andreas/Linux_Data/hpc-semeval/tensorL05add2/out/',
        # 'tensorL05con': '/media/andreas/Linux_Data/hpc-semeval/tensorL05con/out/',
        # 'tensorL05con2': '/media/andreas/Linux_Data/hpc-semeval/tensorL05con2/out/',
        # 'theanoL05add': '/media/andreas/Linux_Data/hpc-semeval/theanoL05add/out/',
        # 'local2': '/media/andreas/Linux_Data/hpc-semeval/local/done2/',
        # 'tensorL05conX': '/media/andreas/Linux_Data/hpc-semeval/tensorL05conX/out/',
        # 'tensorX': '/media/andreas/Linux_Data/hpc-semeval/tensorX/out/',
        # 'all': '/home/andreas/workspace/semeval/project/prob/',
        'tensorL05con2redo2': '/media/andreas/Linux_Data/hpc-semeval/tensorL05con2redo2/out/',
    }

    ser_dev_true = pd.read_csv('../data/dev/dev-full.txt', sep='\t',
                               index_col=0, usecols=[ID, LABEL], header=0, squeeze=True)
    ser_tst_true = pd.read_csv('../data/test/test-full.txt', sep='\t',
                               index_col=0, usecols=[ID, LABEL], header=0, squeeze=True)
    dev_true = ser_dev_true.values.flatten()
    tst_true = ser_tst_true.values.flatten()
    f = np.vectorize(get_conf, otypes=[np.float])

    for k, d in names.items():
        directory = listdir(d)
        filtr = ''
        if filtr == 'soft':
            suffix = r'.*\.[6-7]\d\d\.npy$'
        elif filtr == 'default':
            suffix = r'.*(\.6[7-9]|\.7\d)\d\.npy$'
        elif filtr == 'hard':
            suffix = r'.*(\.69|\.7\d)\d\.npy$'
        else:
            suffix = r'.*\.npy$'

        get_dev_score = lambda x: accuracy_score(y_true=dev_true, y_pred=x)
        get_tst_score = lambda x: accuracy_score(y_true=tst_true, y_pred=x)
        acc_extractor = lambda s: 0.001 * int(s.split('.')[-2])
        vfunc = np.vectorize(acc_extractor)

        new = True
        if not new:
            dev_files = sorted([f for f in directory if re.match(r'^probabilities-dev'+suffix, f)])
            dev_probs = np.asarray([np.load(d + y).flatten() for y in dev_files])
            dev_preds = (dev_probs >= 0.5)
            dev_confs = f(dev_probs.copy())
            dev_conf_scores = []
            dev_major_scores = []
            dev_scores = np.apply_along_axis(get_dev_score, 1, dev_preds)
            dev_mean = np.mean(dev_scores)

            tst_files = sorted([f for f in directory if re.match(r'^probabilities-tst'+suffix, f)])
            tst_probs = np.asarray([np.load(d + y).flatten() for y in tst_files])
            tst_preds = (tst_probs >= 0.5)
            tst_confs = f(tst_probs.copy())
            tst_conf_scores = []
            tst_major_scores = []
            tst_scores = np.apply_along_axis(get_tst_score, 1, tst_preds)
            tst_mean = np.mean(tst_scores)

        if new:
            dev_files = [f for f in directory if re.match(r'^probabilities-dev'+suffix, f)]
            dev_probs = [(y, np.load(d + y).flatten()) for y in dev_files]
            dev_df = pd.DataFrame(dev_probs, columns=['id', 'dev prob'])
            dev_df['id'] = dev_df['id'].map(lambda x: x[18:62].replace(' ', '0'))
            dev_df.set_index('id', inplace=True, drop=True)
            dev_df['dev prob'] = dev_df['dev prob'].map(lambda x: (x,))
            dev_df['dev conf'] = dev_df.apply(lambda row: (f(row['dev prob'][0]),), axis=1)
            dev_df['dev pred'] = dev_df.apply(lambda row: (row['dev prob'][0] >= 0.5,), axis=1)
            dev_df['dev score'] = dev_df.apply(lambda row: get_dev_score(row['dev pred'][0]), axis=1)
            # print(dev_df['dev score'], 5)

            tst_files = [f for f in directory if re.match(r'^probabilities-tst'+suffix, f)]
            tst_probs = [(y, np.load(d + y).flatten()) for y in tst_files]
            tst_df = pd.DataFrame(tst_probs, columns=['id', 'tst prob'])
            tst_df['id'] = tst_df['id'].map(lambda x: x[18:62].replace(' ', '0'))
            tst_df.set_index('id', inplace=True, drop=True)
            tst_df['tst prob'] = tst_df['tst prob'].map(lambda x: (x,))
            tst_df['tst conf'] = tst_df.apply(lambda row: (f(row['tst prob'][0]),), axis=1)
            tst_df['tst pred'] = tst_df.apply(lambda row: (row['tst prob'][0] >= 0.5,), axis=1)
            tst_df['tst score'] = tst_df.apply(lambda row: get_tst_score(row['tst pred'][0]), axis=1)
            # print(tst_df['tst score'], 5)

            df = dev_df.join(tst_df, how='outer')
            df = df.sort_values('dev score')

            np.set_printoptions(linewidth=250)
            tprint(df, 5)

        quit()

        # print more stats
        if False:
            pd.set_option('display.float_format', lambda x: '%.6f' % x)
            print('dev:\n', pd.Series(dev_mean).describe())
            print('test:\n', pd.Series(tst_mean).describe())

        length = range(1, len(dev_files)+1)
        # length = range(1, 100)
        for i in length:
            print(i)
            acc_conf_dev, acc_major_dev = combine(dev_true, y_pred=dev_preds[:i], conf=dev_confs[:i])
            acc_conf_tst, acc_major_tst = combine(tst_true, y_pred=tst_preds[:i], conf=tst_confs[:i])
            dev_conf_scores.append(acc_conf_dev)
            tst_conf_scores.append(acc_conf_tst)
            dev_major_scores.append(acc_major_dev)
            tst_major_scores.append(acc_major_tst)

        mtrx = {
            'dev: confidence vote': dev_conf_scores,
            'test: confidence vote': tst_conf_scores,
            'dev: majority vote': dev_major_scores,
            'test: majority vore': tst_major_scores,
            'dev: mean accuracy': dev_mean,
            'test: mean accuracy': tst_mean,
        }
        df = pd.DataFrame(mtrx)
        plot(df, k + 'all_')


if __name__ == '__main__':
    main()
