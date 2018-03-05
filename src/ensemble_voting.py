""" combining predictions based on votes by a set of answer files """
import pandas as pd
import numpy as np
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from sklearn.metrics import accuracy_score
from os import listdir
import re
import matplotlib.pyplot as plt
from constants import LABEL
from preprocessing import get_train_dev_test
from results_evaluator import tprint
import seaborn as sns


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


def plot_axis(df, ax, legend_pos='orig1'):
    df.plot(x=np.arange(1, len(df) + 1), ax=ax, use_index=False, xlim=[-25, len(df) + 25], ylim=[0.5, 0.775],
            style=['-', '-', '-', '-'], lw=1.5,
            yticks=[.5, .525, .55, .575, .6, .625, .65, .675, .7, .725, .75, .775])
    ax.lines[1].set_linewidth(0.9)  # 1.15
    ax.lines[3].set_linewidth(0.9)  # 1.15
    col1 = ax.lines[0].get_color()
    col2 = ax.lines[2].get_color()
    ax.lines[1].set_color(tuple(1.3*c for c in col1))  # 1.1*
    ax.lines[3].set_color(tuple(1.3*c for c in col2))  # 1.1*

    ax.grid(b=True, which='major', linestyle='-', linewidth=0.85)
    ax.grid(b=True, which='minor', linestyle=':', linewidth=0.75)

    if legend_pos == 'orig1':
        ax.legend(loc='center', bbox_to_anchor=(0.5, 0.365))
    elif legend_pos == 'orig2':
        ax.legend().remove()
    elif legend_pos == 'alt1':
        ax.legend(loc='center', bbox_to_anchor=(0.5, 0.14))
    elif legend_pos == 'alt2':
        ax.legend(loc='lower left', bbox_to_anchor=(0.02, 0))
    else:
        ax.legend()

    ax.set_xlabel('number of models', weight='bold')
    ax.set_ylabel('accuracy', weight='bold')

    # majorLocator_x = MultipleLocator(500)
    majorLocator_y = MultipleLocator(.05)
    majorFormatter_y = FormatStrFormatter('%.2f')
    minorLocator_y = MultipleLocator(.025)

    # ax.xaxis.set_major_locator(majorLocator_x)
    ax.yaxis.set_major_locator(majorLocator_y)
    ax.yaxis.set_major_formatter(majorFormatter_y)

    # for the minor ticks, use no labels; default NullFormatter
    ax.yaxis.set_minor_locator(minorLocator_y)


def plot_figure(dfs: list, name, show=True, save=False, legend_pos: list=None, align='h'):
    length = len(dfs)
    if legend_pos is None:
        legend_pos = [''] * length

    sns.set(color_codes=True, font_scale=1)
    sns.set_style("whitegrid", {'legend.frameon': True})
    sns.set_palette("deep")

    if align == 'h':
        fig, ax = plt.subplots(ncols=length, figsize=(5*length, 5), sharey=True)
    else:
        fig, ax = plt.subplots(nrows=length, figsize=(5, 5*length))

    if length > 1:
        for i, df in enumerate(dfs):
            plot_axis(df, ax[i], legend_pos=legend_pos[i])
        ax[0].set_title('original dataset')
        ax[1].set_title('alternative (randomized) data split')
    else:
        plot_axis(dfs[0], ax, legend_pos=legend_pos[0])

    fig.tight_layout()
    if show:
        plt.show()
    if save:
        fig.savefig(name + '.pdf', bbox_inches='tight')
    plt.close('all')


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
        # 'tensorL05con2redo2': '/media/andreas/Linux_Data/hpc-semeval/tensorL05con2redo2/out/',
        'alt_split_odd': '/media/andreas/Linux_Data/hpc-semeval/alt_split_odd/out/',
    }

    _, df_dev_data, df_tst_data = get_train_dev_test(options=dict(alt_split=True))

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
            # 'dev: confidence vote': dev_conf_scores,
            # 'test: confidence vote': tst_conf_scores,
            'test: mean accuracy': tst_mean,
            'dev: sorted accuracy': df['dev', 'acc', 0],
            'dev: majority vote': dev_major_scores,
            'test: majority vote': tst_major_scores,
            # 'dev: mean accuracy': dev_mean,
        }

        df = pd.DataFrame(mtrx)
        plot_figure([df], k + 'all_')
        # df.to_csv('../figures/alt-split.csv', sep='\t')


if __name__ == '__main__':
    # main()
    df1 = pd.read_csv('../out/orig-split.csv', sep='\t')
    df2 = pd.read_csv('../out/alt-split.csv', sep='\t')
    plot_figure([df1], '../out/orig-split_2', save=True, legend_pos=['orig1'])
    plot_figure([df2], '../out/alt-split_2', save=True, legend_pos=['alt1'])
    plot_figure([df1, df2], '../out/ensemble_h_2', save=True, legend_pos=['orig2', 'alt2'], align='h')
    plot_figure([df1, df2], '../out/ensemble_v_2', save=True, legend_pos=['orig2', 'alt2'], align='v')
