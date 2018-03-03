""" combining predictions based on votes by a set of answer files """
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from os import listdir
import re
import matplotlib.pyplot as plt
from constants import LABEL
from preprocessing import get_train_dev_test
from results_evaluator import tprint


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


def plot(df, name, show=True, save=False):

    tprint(df, 5)

    fig, ax = plt.subplots()
    df.plot(ax=ax)
    ax.yaxis.grid()
    # ax.set_ylim(.5, .8)
    ax.set_xlabel('number of models')
    ax.set_ylabel('accuracy')
    ax.set_title('Ensemble scores')

    # ax2 = ax1.twiny()

    # xtick_range = np.arange(0, len(df), len(df)//16)
    # xticks = x.iloc[xtick_range]
    # xticks = np.round(xticks, 3)
    # ax2.set_xticks(xtick_range)
    # ax2.set_xticklabels(xticks)

    if show:
        plt.show()
    if save:
        fig.savefig('../figures/' + name + '.pdf')


def build_df(files, y_true):
    probs_ser_lst = [pd.Series(np.load(f).flatten(), name=f[-48:-4].replace(' ', '0')) for f in files]
    probs_df = pd.DataFrame(probs_ser_lst)
    preds_df = probs_df.applymap(lambda x: x >= 0.5)
    confs_df = probs_df.apply(lambda x: np.abs(x - 0.5))
    accs_ser = preds_df.apply(lambda row: accuracy_score(y_true=y_true, y_pred=row), axis=1)
    df = pd.concat([accs_ser, preds_df, probs_df, confs_df], axis=1,
                   keys=['acc', 'pred', 'prob', 'conf'])
    return df


def main():
    names = {
        'tensorL05con2redo2': '/media/andreas/Linux_Data/hpc-semeval/tensorL05con2redo2/out/',
        # 'alt_split_odd': '/media/andreas/Linux_Data/hpc-semeval/alt_split_odd/out/',
    }

    _, df_dev_data, df_tst_data = get_train_dev_test(options=dict(alt_split=False))

    dev_true = df_dev_data[LABEL].values.flatten()
    tst_true = df_tst_data[LABEL].values.flatten()

    for k, d in names.items():
        directory = listdir(d)
        dev_files = [d + f for f in directory if re.match(r'^probabilities-' + 'dev', f)]
        tst_files = [d + f for f in directory if re.match(r'^probabilities-' + 'tst', f)]

        df_dev = build_df(dev_files, dev_true)
        df_tst = build_df(tst_files, tst_true)

        df = pd.concat([df_dev, df_tst], axis=1, keys=['dev', 'tst'])
        df = df.sort_values(('dev', 'acc', 0), ascending=False)

        dev_acc_filter = 0.
        if dev_acc_filter:
            row_filter = df['dev', 'acc', 0] >= dev_acc_filter
            df = df[row_filter.values]
            print('filtered for dev accuracies >=', dev_acc_filter)

        dev_mean = np.mean(df['dev', 'acc', 0].values)
        tst_mean = np.mean(df['tst', 'acc', 0].values)

        dev_preds_np = df['dev', 'pred'].values
        dev_confs_np = df['dev', 'conf'].values
        tst_preds_np = df['tst', 'pred'].values
        tst_confs_np = df['tst', 'conf'].values

        # print more stats
        if False:
            pd.set_option('display.float_format', lambda x: '%.6f' % x)
            print('dev:\n', pd.Series(dev_mean).describe())
            print('test:\n', pd.Series(tst_mean).describe())

        dev_conf_scores = list()
        tst_conf_scores = list()
        dev_major_scores = list()
        tst_major_scores = list()

        length = range(1, len(df)+1)
        # length = range(1, 100)
        for i in length:
            acc_conf_dev, acc_major_dev = combine(dev_true, y_pred=dev_preds_np[:i], conf=dev_confs_np[:i])
            acc_conf_tst, acc_major_tst = combine(tst_true, y_pred=tst_preds_np[:i], conf=tst_confs_np[:i])
            dev_conf_scores.append(acc_conf_dev)
            tst_conf_scores.append(acc_conf_tst)
            dev_major_scores.append(acc_major_dev)
            tst_major_scores.append(acc_major_tst)

        mtrx = {
            'dev: confidence vote': dev_conf_scores,
            'test: confidence vote': tst_conf_scores,
            'dev: majority vote': dev_major_scores,
            'test: majority vore': tst_major_scores,
            # 'dev: mean accuracy': dev_mean,
            'test: mean accuracy': tst_mean,
            'dev: sorted accuracy': df['dev', 'acc', 0],
        }

        df = pd.DataFrame(mtrx)
        plot(df, k + 'all_')


if __name__ == '__main__':
    main()
