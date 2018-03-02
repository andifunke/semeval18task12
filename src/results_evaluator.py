import random
import re
from os import listdir
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pylab, gridspec
from matplotlib.ticker import LinearLocator
from tabulate import tabulate
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import numpy as np
import seaborn as sns

# making stuff more human readable
AXES = {
    'dev_acc': "Dev Accuracy",
}
LABELS = {
    'cb': "CBOW",
}
TITLES = {
    'embedding_model': "Word2Vec model",
}
# and adding some fancy colors for the scatter plots
INDEXES = {
    'embedding_size': (12, 25, 50, 100),
    'embedding_model': ('sg', 'cb'),
    'max_iter': (-1, 1000),
    'scale': (False, True),
    'test_size': (2000000, 4853410),
    'train_size': (100000, 400000, 888237),
    'C': (1, 2, 1000),
    'lowercase': (False, True),
    'shrinking': (False, True),
}
COLORS = ('red', 'green', 'blue', 'yellow', 'orange', 'violet')
MARKERS = ('o', '^', 's', 'p')

# paths
SUB_DIR = 'tensorL05con2redo2/'
DAT_DIR = './results/' + SUB_DIR
FIG_DIR = './figures/' + SUB_DIR
KEYS = [
    'timestamp',
    'dev_acc',
    'val_acc',
    'test_acc',
    'epoch',
    'backend',
    'classifier',
    'dimensionality',
    'padding',
    'lstm_size',
    'activation1',
    'activation2',
    'optimizer',
    'loss',
    'batch_size',
    'dropout',
    'vsplit',
    'embedding',
    'embedding2',
    'rich',
    'runtime',
    'alt_split',
    'dev_test_ratio',
    # 'epochs',
    # 'pre_seed',
    'run',
    # 'run seed',
    # 'runs',
    # 'vocabulary',
    # 'words in embeddings',
    'argv',
]
DEF_KEYS = {
    'epoch': False,
    'backend': False,
    'classifier': True,
    # 'dimensionality': None,
    'padding': True,
    'lstm_size': True,
    'activation1': True,
    # 'activation2': True,
    'optimizer': True,
    'loss': True,
    'batch_size': True,
    'dropout': True,
    'vsplit': True,
    'embedding': True,
    # 'embedding2': True,
    # 'rich': True,
    'alt_split': True,
    'dev_test_ratio': True,
}
LIM = .45, .75


def tprint(df: pd.DataFrame, head=0, to_latex=False):
    if head > 0:
        df = df.head(head)
    elif head < 0:
        df = df.tail(-head)
    print(tabulate(df, headers="keys", tablefmt="pipe", floatfmt=".3f") + '\n')

    if to_latex:
        print(df.to_latex(bold_rows=True))


def plot_group(df, columns, x='epoch', y='dev_acc', ylim=(0, 1), interval=None, show=False, log=False):
    """
    creates a scatter plot for a given key usually with the train/test time on the x axis
    and the f1_micro measure on the y axis. You can give in two keys at the same time
    (for example combined with the train_size) which will plot with a different marker shape
    for the second key.
    """
    fname = FIG_DIR + "{}__{}__{}".format('_'.join(columns), x, y)
    multidim = len(columns) > 1

    fig, ax = plt.subplots()
    for key, group in df.groupby(columns):
        # define the label for the legend
        if multidim:
            label1 = LABELS[key[0]] if key[0] in LABELS else str(key[0])
            column2 = LABELS[columns[1]] if columns[1] in LABELS else str(columns[1])
            label2 = LABELS[key[1]] if key[1] in LABELS else str(key[1])
            label = "{} | {}: {}".format(label1, column2, label2)
        else:
            label = LABELS[key] if key in LABELS else key

        # plot each group
        if multidim:
            color = COLORS[INDEXES[columns[0]].index(key[0])]
            marker = MARKERS[INDEXES[columns[1]].index(key[1])]
        else:
            if isinstance(key, (float, int)):
                cmap = cm.tab20
                if interval is not None:
                    norm = Normalize(vmin=interval[0], vmax=interval[1])
                    color = cmap(norm(key))
                else:
                    color = cmap(key)
            else:
                color = COLORS[INDEXES[columns[0]].index(key)]
            marker = None
        group.plot.scatter(ax=ax, x=x, y=y, label=label, color=color, marker=marker, s=7)

    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(TITLES[columns[0]] if columns[0] in TITLES else columns[0])
    plt.grid()

    print("saving", fname)
    ax.legend(bbox_to_anchor=(0.98, 1.05))
    ax.set_autoscaley_on(False)
    ax.set_ylim(ylim)
    if log:
        # saving also a log scaled x-axis
        ax.set_xscale('log')
    fig.savefig(fname + ".pdf")
    if show:
        plt.show()
    plt.clf()
    plt.close()


def boxplot_group(df, x='epoch', y='dev_acc', ylim=None, show=False):
    fname = FIG_DIR + "{}__{}__{}".format('box', x, y)
    df = df[[y, x]]
    print(x, df[x].dtype)
    if df[x].dtype != object:
        df[x] = df[x].astype(str)
    group = df.groupby(x)
    fig, ax = plt.subplots()
    group.boxplot(ax=ax)

    # not working:
    ax.get_yaxis().set_major_locator(LinearLocator(numticks=20))
    if ylim is not None:
        ax.set_autoscaley_on(False)
        ax.set_ylim(ylim)

    if show:
        plt.show()
    print("saving", fname)
    fig.savefig(fname + ".pdf")
    plt.clf()
    plt.close()


def aggregate_group(dfs, df_descriptions, keys, to_latex=False):
    """
    dfs can be a single DataFrame or list/tuple of DataFrame.
    prints out some interesting aggregations for given keys.
    """
    print('********************************************************************************************************')
    print('AGGREGATIONS FOR KEYED GROUP:', keys)
    print()

    main_metric = 'dev_acc'

    if isinstance(dfs, pd.DataFrame):
        df_list = [dfs]
    else:
        df_list = dfs

    for df, df_d in zip(df_list, df_descriptions):
        print("on data set: ", df_d)

        # filter if appropriate
        argv_filter = [key for key in keys if DEF_KEYS[key]]
        if len(argv_filter) > 0:
            print("filter:", argv_filter)
            df = df[df.argv.str.contains('|'.join(argv_filter), na=False)]

        group = df.groupby(keys)

        print("row of maximum for {} per group:".format(main_metric))
        # get the row with the maximum value of 'f1_micro' per group
        df_f1micmax_for_group = df.loc[group[main_metric].idxmax()]
        # reset the index
        df_f1micmax_for_group.set_index(keys, inplace=True)
        tprint(df_f1micmax_for_group.sort_values(main_metric, ascending=False), to_latex=to_latex)

        # to avoid missing keys
        columns_full = [
            main_metric,
        ]
        columns_reduced = []
        for column in columns_full:
            if column not in keys:
                columns_reduced.append(column)

        if False:
            print("maximum of all columns - sorted by {}:".format(main_metric))
            df_max = group.max()[columns_reduced].sort_values(main_metric, ascending=False)
            tprint(df_max, to_latex=to_latex)

        print("mean of all columns - sorted by {}:".format(main_metric))
        tprint(group.mean()[columns_reduced].sort_values(main_metric, ascending=False), to_latex=to_latex)

    print('********************************************************************************************************')


def load_results(directory=None):
    dat_dir = DAT_DIR if directory is None else directory
    files = [f for f in listdir(dat_dir) if re.match(r'^report_.*\.csv$', f)]

    # loading test-results
    results = []
    for fname in files:
        result = pd.read_csv(dat_dir + fname, sep='\t', index_col=None,
                             converters={'classifier': (lambda arg: 'LSTM_01' if arg == 'AttentionLSTM' else arg)})
        result.rename(columns={'pred_acc': 'dev_acc'}, inplace=True)
        results.append(result)
        print(fname)
        # tprint(result)

    df = pd.concat(results, ignore_index=True)
    if 'test_acc' not in df.keys():
        df['test_acc'] = np.nan

    return df


def plot1(df: pd.DataFrame):
    df = df[['val_acc', 'dev_acc', 'test_acc']].applymap(lambda x: x + random.uniform(-0.001, 0.001))
    fig, ax = plt.subplots(ncols=3)

    df.plot(ax=ax[0], x='dev_acc', y='val_acc', kind='scatter', s=2)
    ax[0].set_xlim(LIM)
    ax[0].set_ylim(LIM)
    ax[0].set_xlabel('accuracy score: dev')
    ax[0].set_ylabel('accuracy score: val')

    df.plot(ax=ax[1], x='dev_acc', y='test_acc', kind='scatter', s=2)
    ax[1].set_xlim(LIM)
    ax[1].set_ylim(LIM)
    ax[1].set_xlabel('accuracy score: dev')
    ax[1].set_ylabel('accuracy score: test')

    df.plot(ax=ax[2], x='val_acc', y='test_acc', kind='scatter', s=2)
    ax[2].set_xlim(LIM)
    ax[2].set_ylim(LIM)
    ax[2].set_xlabel('accuracy score: val')
    ax[2].set_ylabel('accuracy score: test')
    plt.show()

    fig, ax = plt.subplots(nrows=2)
    df.plot(ax=ax[0], kind='density')
    df.plot(ax=ax[1], kind='box')
    plt.show()
    plt.close('all')


def plot2(df):
    data = df
    ylim = [0.4, 0.8]
    plot_group(data, ['epoch'], x='dropout', interval=(1, 20), ylim=ylim)
    plot_group(data, ['epoch'], x='epoch', interval=(1, 20), ylim=ylim)
    plot_group(data, ['epoch'], x='padding', interval=(1, 20), ylim=ylim)
    plot_group(data, ['epoch'], x='lstm_size', interval=(1, 20), ylim=ylim, log=True)
    plot_group(data, ['epoch'], x='batch_size', interval=(1, 20), ylim=ylim, log=True)
    plot_group(data, ['epoch'], x='vsplit', interval=(1, 20), ylim=ylim)

    boxplot_group(data, x='backend', ylim=ylim)
    boxplot_group(data, x='dropout')
    boxplot_group(data, x='epoch')
    boxplot_group(data, x='padding')
    boxplot_group(data, x='lstm_size')
    boxplot_group(data, x='batch_size')
    boxplot_group(data, x='vsplit')
    boxplot_group(data, x='rich')
    boxplot_group(data, x='rich')
    boxplot_group(data, x='activation1')
    boxplot_group(data, x='optimizer')
    boxplot_group(data, x='loss')
    boxplot_group(data, x='embedding')
    plt.close('all')


def plot3(df_, save=False, single_plot: bool=False, directory='', fname=''):
    df = df_.copy(deep=True)

    sns.set(color_codes=True, font_scale=1)
    sns.set_style("whitegrid", {'legend.frameon': True})
    df.rename(columns={'dev_acc': 'dev', 'test_acc': 'test', 'val_acc': 'cross validation'}, inplace=True)
    # adding some jitter for a less quantized plot
    df = df[['cross validation', 'dev', 'test']].applymap(lambda x: x + random.uniform(-0.0005, 0.0005))

    def single_scatter(data, x_key, y_key, axis, xlim=LIM, ylim=LIM, xticks=None, yticks=None):
        axis.set_xlim(xlim)
        axis.set_ylim(ylim)
        if xticks is not None:
            axis.set_xticks(xticks)
        if yticks is not None:
            axis.set_yticks(yticks)
        sns.regplot(x=data[x_key], y=data[y_key], ax=axis, scatter=True, fit_reg=False, scatter_kws={'s': 4})
        axis.set_xlabel('accuracy score: ' + x_key, weight='bold', size=11)
        axis.set_ylabel('accuracy score: ' + y_key, weight='bold', size=11)

    if single_plot:
        # single scatter plot dev/test
        fig, ax = plt.subplots(figsize=(5, 3.3))
        single_scatter(df, 'dev', 'test', ax, ylim=(.45, .65), yticks=[.45, .5, .55, .6, .65])
        fig.tight_layout()
        if save:
            fig.savefig(directory + fname + '_dev-test_scatter.pdf', bbox_inches='tight')
        plt.show()
        plt.close('all')

    # multi-plot
    gs = gridspec.GridSpec(2, 6)
    fig = plt.figure(figsize=(15, 10))
    ax0_0 = fig.add_subplot(gs[0, :2])
    ax0_1 = fig.add_subplot(gs[0, 2:4])
    ax0_2 = fig.add_subplot(gs[0, 4:])
    ax1_0 = fig.add_subplot(gs[1, :3])
    ax1_1 = fig.add_subplot(gs[1, 3:])

    # scatter plot for dev/test, val/test, dev/val
    single_scatter(df, 'dev', 'test', ax0_0)
    single_scatter(df, 'cross validation', 'test', ax0_1)
    single_scatter(df, 'dev', 'cross validation', ax0_2)

    # plot distributions
    sns.distplot(df['dev'], ax=ax1_0, bins=5, rug=False, kde=True, label='dev')
    sns.distplot(df['test'], ax=ax1_0, bins=5, rug=False, kde=True, label='test')
    sns.distplot(df['cross validation'], ax=ax1_0, bins=5, rug=False, kde=True, label='cross validation')
    ax1_0.set_xlim(LIM)
    ax1_0.set_xlabel('distribution of accuracy scores', weight='bold')
    ax1_0.legend()

    sns.boxplot(data=df[['dev', 'test', 'cross validation']], ax=ax1_1, notch=True, width=.3)
    ax1_1.set_xlabel('distribution of accuracy scores', weight='bold')
    ax1_1.set_ylim(LIM)

    fig.tight_layout()
    if save:
        fig.savefig(directory + fname + '_distributions.pdf', bbox_inches='tight')
    plt.show()
    plt.close('all')


def print_stats(df, by_group=False, write=False, directory='', fname=''):
    """ get statistics over all runs """
    df = df[['val_acc', 'dev_acc', 'test_acc']]
    if by_group:
        df_stats = df.groupby('timestamp').describe()
        print(df_stats.sort_values(('dev_acc', 'mean'), ascending=False))
    else:
        df_stats = df.describe()
        print(df_stats)

    if write:
        fname = directory + fname + '_stats.csv'
        df_stats.to_csv(fname, sep='\t')


def print_metrics(dfs, additional_keys=None):
    if additional_keys is None:
        additional_keys = []
    dfs_descriptions = ["full data set"]
    for c in DEF_KEYS:
        aggregate_group(dfs, dfs_descriptions, [c] + additional_keys)


def print_results(df, write=False, directory=None, fname=''):
    tprint(df.sort_values('dev_acc', ascending=False))

    if write:
        fname = directory + fname + '_metrics.csv'
        df.to_csv(fname, sep='\t')


def reports_evaluator_main(in_directory=None, out_directory='../out/', fname=''):
    """
    a given directory overwrites the defaults. The function will look for all test-result files in this directory.
    """
    print('start evaluating')
    df = load_results(in_directory)[KEYS]

    if False:
        # filter for best epoch
        keys = ['timestamp', 'run']
        group = df.groupby(keys)
        # get the row with the maximum value of 'dev_acc' per group
        df = df.loc[group['dev_acc'].idxmax()]
        # reset the index
        df.set_index(keys, inplace=True)

    # print grouped tables
    print_results(df)
    # print_metrics(df)
    print_stats(df, write=True, directory=out_directory, fname=fname)
    plot3(df, save=True, directory=out_directory, fname=fname)


if __name__ == '__main__':
    # reports_evaluator_main('/media/andreas/Linux_Data/hpc-semeval/tensorL05con2redo2/out/',
    #                        fname='results_2560-single_orig-split_rescaled')
    reports_evaluator_main('../out/',
                           fname='test')
