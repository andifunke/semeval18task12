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


def combine(series, list_of_files, d):
    """ data loading is highly inefficient ^^ """
    y_true = series.values.flatten()
    y_prob = np.asarray([np.load(d + y).flatten() for y in list_of_files])
    length = len(y_prob[0])
    y_pred = (y_prob >= 0.5)

    """ confidence vote """
    prob = np.full(length, 0.5, dtype=np.float64)
    for prob_tmp in y_prob:
        conf = get_conf(prob)
        conf_tmp = get_conf(prob_tmp)
        vote_tmp = conf_tmp > conf
        prob[vote_tmp] = prob_tmp[vote_tmp]
        pred = prob >= 0.5
        acc_conf = accuracy_score(y_true=y_true, y_pred=pred)
    print('combined accuracy by confidence: {:.3f}'.format(acc_conf))

    """ majority vote """
    pred_conf = pred
    pred = np.mean(y_pred, axis=0)
    # in case of a tie use the predictions from the condifdence vote
    tie = np.isclose(pred, 0.5)
    pred[tie] = pred_conf[tie]
    pred = (pred >= 0.5)
    acc_major = accuracy_score(y_true=y_true, y_pred=pred)
    print('combined accuracy by majority vote: {:.3f}'.format(acc_major))

    series[:] = pred
    return 1 * series, acc_conf, acc_major


def plot(df, name, show=True):
    fig = plt.figure()
    df.plot()
    plt.xlabel('number of models')
    plt.ylabel('accuracy')
    plt.title('Ensemble scores')
    if show:
        plt.show()
    fig.savefig('./figures/' + name + '.pdf')


def main():
    ser_dev_true = pd.read_csv('./data/dev/dev-only-labels.txt', sep='\t', index_col=0, header=0, squeeze=True)
    ser_tst_zero = pd.read_csv('./data/gold/truth.txt', sep='\t', index_col=0, header=0, squeeze=True)

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
    for k, d in names.items():
        y_dev_files = sorted([f for f in listdir(d) if re.match(r'^probabilities-dev_.*\.npy$', f)])
        y_tst_files = sorted([f for f in listdir(d) if re.match(r'^probabilities-tst_.*\.npy$', f)])

        y_axis_conf_dev = []
        y_axis_major_dev = []
        y_axis_conf_tst = []
        y_axis_major_tst = []
        length = range(1, len(y_dev_files)+1)
        # length = range(1, 100)
        for i in length:
            print(i)
            dev_acc = float(y_dev_files[i-1][-9:-4])
            if False and dev_acc < 0.67:
                continue
            ser_dev_pred, acc_conf_dev, acc_major_dev = combine(ser_dev_true.copy(), y_dev_files[:i], d)
            ser_tst_pred, acc_conf_tst, acc_major_tst = combine(ser_tst_zero.copy(), y_tst_files[:i], d)
            y_axis_conf_dev.append(acc_conf_dev)
            y_axis_conf_tst.append(acc_conf_tst)
            y_axis_major_dev.append(acc_major_dev)
            y_axis_major_tst.append(acc_major_tst)

        mtrx = {
            'dev: confidence vote': y_axis_conf_dev,
            'test: confidence vote': y_axis_conf_tst,
            'dev: majority vote': y_axis_major_dev,
            'test: majority vore': y_axis_major_tst,
        }
        df = pd.DataFrame(mtrx)
        tprint(df)
        df.to_csv('/media/andreas/Linux_Data/hpc-semeval/{}_all_.csv'.format(k), sep='\t')
        plot(df, k + 'all_')


if __name__ == '__main__':
    main()
