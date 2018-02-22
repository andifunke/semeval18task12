""" combining predictions based on votes by a set of answer files """
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from os import listdir
import re
import matplotlib.pyplot as plt
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

    ser_dev_true = pd.read_csv('./data/dev/dev-only-labels.txt', sep='\t', index_col=0, header=0, squeeze=True)
    ser_tst_true = pd.read_csv('./data/gold/truth.txt', sep='\t', index_col=0, header=0, squeeze=True)
    dev_true = ser_dev_true.values.flatten()
    tst_true = ser_tst_true.values.flatten()
    f = np.vectorize(get_conf, otypes=[np.float])

    for k, d in names.items():
        directory = listdir(d)
        filtr = 'soft'
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

        dev_files = sorted([f for f in directory if re.match(r'^probabilities-dev'+suffix, f)])
        dev_probs = np.asarray([np.load(d + y).flatten() for y in dev_files])
        dev_preds = (dev_probs >= 0.5)
        dev_confs = f(dev_probs.copy())
        dev_conf_scores = []
        dev_major_scores = []
        dev_scores = np.apply_along_axis(get_dev_score, 1, dev_preds)
        dev_scores[:] = np.mean(dev_scores)

        tst_files = sorted([f for f in directory if re.match(r'^probabilities-tst'+suffix, f)])
        tst_probs = np.asarray([np.load(d + y).flatten() for y in tst_files])
        tst_preds = (tst_probs >= 0.5)
        tst_confs = f(tst_probs.copy())
        tst_conf_scores = []
        tst_major_scores = []
        tst_scores = np.apply_along_axis(get_tst_score, 1, tst_preds)
        tst_scores[:] = np.mean(tst_scores)

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
            'dev: mean accuracy': dev_scores,
            'test: mean accuracy': tst_scores,
        }
        df = pd.DataFrame(mtrx)
        plot(df, k + 'all_')


if __name__ == '__main__':
    main()
