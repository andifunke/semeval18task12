import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from os import listdir
import re

np.set_printoptions(linewidth=8000)


def get_conf(prob):
    return np.abs(prob - 0.5)


def combine(series, list_of_files, name, d):
    y_true = series.values.flatten()

    y_prob = np.asarray([np.load(d + y).flatten() for y in list_of_files])
    length = len(y_prob[0])

    y_pred = (y_prob >= 0.5)

    for y_p in y_pred:
        acc = accuracy_score(y_true=y_true, y_pred=y_p)
        # print('accuracy: {:.3f}'.format(acc))

    """ confidence vote """
    prob = np.full(length, 0.5, dtype=np.float64)
    for prob_tmp in y_prob:
        conf = get_conf(prob)
        conf_tmp = get_conf(prob_tmp)
        vote_tmp = conf_tmp > conf
        prob[vote_tmp] = prob_tmp[vote_tmp]
        pred = prob >= 0.5
        acc = accuracy_score(y_true=y_true, y_pred=pred)

    print('combined accuracy by confidence: {:.3f}'.format(acc))

    """ majority vote """
    pred_conf = pred
    # pred = np.zeros(length)
    pred = np.mean(y_pred, axis=0)
    # in case of a tie use the predictions from the condifdence vote
    tie = np.isclose(pred, 0.5)
    pred[tie] = pred_conf[tie]
    pred = (pred >= 0.5)
    acc = accuracy_score(y_true=y_true, y_pred=pred)
    print('combined accuracy by majority vote: {:.3f}'.format(acc))

    series[:] = pred
    return 1 * series


def main():
    ser_dev = pd.read_csv('./data/dev/dev-only-labels.txt', sep='\t', index_col=0, header=0, squeeze=True)
    ser_tst = pd.read_csv('./data/test/test-only-data.txt', sep='\t', index_col=0, header=0)
    ser_tst['correctLabelW0orW1'] = 0
    ser_tst = ser_tst['correctLabelW0orW1']

    # d = '/media/andreas/Linux_Data/hpc-semeval-failed/out/'
    # d = '/media/andreas/Linux_Data/hpc-semeval-failed/tensorL05add/out/'
    # d = '/media/andreas/Linux_Data/hpc-semeval-failed/tensorL05add2/out/'
    # d = '/media/andreas/Linux_Data/hpc-semeval-failed/tensorL05con/out/'
    d = '/media/andreas/Linux_Data/hpc-semeval-failed/tensorL05con2/out/'
    name = 'tensorL05con2'
    # d = '/media/andreas/Linux_Data/hpc-semeval-failed/theanoL05add/out/'
    # d = './prob/'

    y_dev_files = [f for f in listdir(d) if re.match(r'^probabilities-dev_.*\.npy$', f)]
    y_tst_files = [f for f in listdir(d) if re.match(r'^probabilities-tst_.*\.npy$', f)]

    ser_dev = combine(ser_dev, y_dev_files, name, d)
    ser_tst = combine(ser_tst, y_tst_files, name, d)

    ser_dev.to_csv('answer_dev_ensemble_{}.txt'.format(name), sep='\t', header=True)
    ser_tst.to_csv('answer_test_ensemble_{}.txt'.format(name), sep='\t', header=True)


if __name__ == '__main__':
    main()
