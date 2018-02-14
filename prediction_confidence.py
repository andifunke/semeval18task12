import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from os import listdir
import re
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=8000)


def get_conf(prob):
    return np.abs(prob - 0.5)


def combine(series, list_of_files, d):
    y_true = series.values.flatten()
    # print(y_true)

    y_prob = np.asarray([np.load(d + y).flatten() for y in list_of_files])
    length = len(y_prob[0])

    y_pred = (y_prob >= 0.5)

    # print(list_of_files)
    # print(y_pred)
    # y_pred = 1 * y_pred
    # print(y_pred)
    # for y_p in y_pred:
    #     # print('pred:', y_p)
    #     # print('true:', y_true)
    #     acc = accuracy_score(y_true=y_true, y_pred=y_p)
    #     print('accuracy: {:.3f}'.format(acc))

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
    # pred = np.zeros(length)
    pred = np.mean(y_pred, axis=0)
    # in case of a tie use the predictions from the condifdence vote
    tie = np.isclose(pred, 0.5)
    pred[tie] = pred_conf[tie]
    pred = (pred >= 0.5)
    acc_major = accuracy_score(y_true=y_true, y_pred=pred)
    print('combined accuracy by majority vote: {:.3f}'.format(acc_major))

    series[:] = pred
    return 1 * series, acc_conf, acc_major


def main():
    ser_dev_true = pd.read_csv('./data/dev/dev-only-labels.txt', sep='\t', index_col=0, header=0, squeeze=True)
    ser_tst_zero = pd.read_csv('./data/test/test-only-data.txt', sep='\t', index_col=0, header=0)
    ser_tst_zero['correctLabelW0orW1'] = 0
    ser_tst_zero = ser_tst_zero['correctLabelW0orW1']

    names = {
        'earlier_models': '/media/andreas/Linux_Data/hpc-semeval/out/',
        'tensorL05add': '/media/andreas/Linux_Data/hpc-semeval/tensorL05add/out/',
        'tensorL05add2': '/media/andreas/Linux_Data/hpc-semeval/tensorL05add2/out/',
        'tensorL05con': '/media/andreas/Linux_Data/hpc-semeval/tensorL05con/out/',
        'tensorL05con2': '/media/andreas/Linux_Data/hpc-semeval/tensorL05con2/out/',
        'theanoL05add': '/media/andreas/Linux_Data/hpc-semeval/theanoL05add/out/',
        'local2': '/media/andreas/Linux_Data/hpc-semeval/local/done2/',
        'tensorL05conX': '/media/andreas/Linux_Data/hpc-semeval/tensorL05conX/out/',
        'tensorX': '/media/andreas/Linux_Data/hpc-semeval/tensorX/out/',
        'all': '/home/andreas/workspace/semeval/project/prob/',
    }
    for k, d in names.items():
        if k != 'all':
            continue
        # k = 'earlier_models'
        # d = names[k]

        y_dev_files = [f for f in listdir(d) if re.match(r'^probabilities-dev_.*\.npy$', f)]
        # y_tst_files = [f for f in listdir(d) if re.match(r'^probabilities-tst_.*\.npy$', f)]

        y_axis_conf = []
        y_axis_major = []
        x_axis = range(1, len(y_dev_files)+1)  # range(1, 3)  #
        for i in x_axis:
            print(i)
            ser_dev_pred, acc_conf, acc_major = combine(ser_dev_true.copy(), y_dev_files[:i], d)
            # ser_tst_pred, _, __ = combine(ser_tst_zero, y_tst_files, d)
            y_axis_conf.append(acc_conf)
            y_axis_major.append(acc_major)

        # print(x_axis)
        # print(y_axis_conf)
        # print(y_axis_major)
        # plt.plot(x_axis, y_axis_conf)
        fig = plt.figure()
        plt.plot(x_axis, y_axis_major)
        plt.xlabel('number of models')
        plt.ylabel('accuracy')
        plt.title('Ensemble scores')
        # plt.savefig('./figures/' + k + '.pdf')
        fig.savefig('./figures/' + k + '.pdf')
        plt.show()

        # ser_dev_pred.to_csv('answer_dev_ensemble_{}.txt'.format(name), sep='\t', header=True)
        # ser_tst_pred.to_csv('answer_test_ensemble_{}.txt'.format(name), sep='\t', header=True)


if __name__ == '__main__':
    main()
