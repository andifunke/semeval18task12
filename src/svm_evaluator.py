""" compare the accuracy distribution of dev and test validation from the SVM models """
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DAT_DIR = '../out/'
FILE = 'svm_results_orig-split'
COLORS = ['navy', 'darkmagenta', 'r', 'm', 'b', 'c', 'y', 'g', 'k', 'w']
ACC_COLS = ['dev', 'test']

LIM_1 = (.45, .650)
TICKS_1 = [0.45, 0.5, 0.55, 0.6, 0.65]
LIM_2 = (.45, .651)
TICKS_2 = [0.45, 0.475, 0.5, 0.525, 0.55, 0.575, 0.6, 0.625, 0.65]
LABEL_SIZE = 10.5

sns.set(color_codes=True, font_scale=0.9)
sns.set_style("whitegrid", {'legend.frameon': True})


def single_scatter(data, x_key, y_key, axis, xlim=LIM_1, ylim=LIM_1, xticks=TICKS_2, yticks=TICKS_2,
                   highlight_scaled=False):
    axis.set_xlim(xlim)
    axis.set_ylim(ylim)
    axis.set_xticks(xticks)
    axis.set_yticks(yticks)
    if highlight_scaled:
        no_scale = data.loc[~data.scale]
        scale = data.loc[data.scale]
        sns.regplot(x=no_scale[x_key], y=no_scale[y_key], ax=axis, scatter=True, fit_reg=False, color=COLORS[0],
                    label='original embeddings')
        sns.regplot(x=scale[x_key], y=scale[y_key], ax=axis, scatter=True, fit_reg=False, color=COLORS[1],
                    label='scaled embeddings')
        axis.legend(loc='lower right')
    else:
        sns.regplot(x=data[x_key], y=data[y_key], ax=axis, scatter=True, fit_reg=False)
    axis.set_xlabel('accuracy score: ' + x_key, weight='bold', size=LABEL_SIZE)
    axis.set_ylabel('accuracy score: ' + y_key, weight='bold', size=LABEL_SIZE)


def plot(df, add_jitter=False, save=False):
    # this takes the data points a little bit off the grid and enhances the plots
    if add_jitter:
        df[ACC_COLS] = df[ACC_COLS].applymap(lambda x: x + random.uniform(-0.0005, 0.0005))

    # --- single scatter plot with scaled not highlighted ---
    fig, ax = plt.subplots(figsize=(5, 3.75))
    single_scatter(df, 'dev', 'test', ax, xlim=[.475, .65], xticks=TICKS_2[1:],
                   ylim=[.475, .625], yticks=TICKS_2[1:-1])
    fig.tight_layout()
    if save:
        fig.savefig(DAT_DIR + FILE + '_scatter.pdf', bbox_inches='tight')
    plt.show()

    quit()
    # --- single scatter plot with scaled highlighted ---
    fig, ax = plt.subplots(figsize=(5, 3.75))
    single_scatter(df, 'dev', 'test', ax, ylim=[.475, .625], yticks=TICKS_2[1:-1], highlight_scaled=True)
    fig.tight_layout()
    fig.savefig(DAT_DIR + FILE + '_scatter1.pdf', bbox_inches='tight')
    plt.show()

    # --- 3x1 plot over accuracy scores distribution ---
    fig, ax = plt.subplots(ncols=3, figsize=(15, 5))
    # ax 1
    single_scatter(df, 'dev', 'test', ax[0], highlight_scaled=True)
    # ax 2
    sns.distplot(df['dev'], ax=ax[1], bins=5, rug=False, kde=True, label='dev')
    sns.distplot(df['test'], ax=ax[1], bins=5, rug=False, kde=True, label='test')
    ax[1].set_xlim(LIM_1)
    ax[1].set_xlabel('distribution of accuracy scores', weight='bold', size=LABEL_SIZE)
    ax[1].legend()
    # ax 3
    sns.boxplot(data=df[ACC_COLS], ax=ax[2], notch=True, width=.3)
    ax[2].set_xlabel('distribution of accuracy scores', weight='bold', size=LABEL_SIZE)
    ax[2].set_ylim(LIM_1)
    # plot
    fig.tight_layout()
    if save:
        fig.savefig(DAT_DIR + FILE + '.pdf', bbox_inches='tight')
    plt.show()

    # --- pairwise parameter plot ---
    df.rename(columns={'dev': 'dev accuracy', 'test': 'test accuracy', 'dims': 'dimensions', 'scale': 'scaled'}, inplace=True)
    # df['dimensions'] = df['dimensions'].astype(str)
    g = sns.pairplot(df, y_vars=['dev accuracy', 'test accuracy'],
                     x_vars=['dev accuracy', 'test accuracy', 'embedding', 'dimensions', 'kernel', 'scaled', 'C'])
    for x in g.axes[:2]:
        for y in x[:2]:
            y.set_xlim(LIM_2)
            y.set_ylim(LIM_1)
            y.set_xticks(TICKS_1)
            y.set_yticks(TICKS_1)
    for x in g.axes:
        for y in x:
            y.tick_params(axis='both', which='major', labelsize=7)

    # embedding
    g.axes[1][2].set_xticklabels([1, 2, 3, 4, 5, 6])
    # dimensions
    g.axes[1][3].set(xscale='log')
    g.axes[1][3].set_xticks([25, 50, 100, 300])
    g.axes[1][3].set_xticklabels([25, 50, 100, 300])
    # scaled
    g.axes[1][5].set_xlim([-0.5, 1.5])
    g.axes[1][5].set_xticks([0, 1])
    g.axes[1][5].set_xticklabels(['False', 'True'])
    # C
    g.axes[1][6].set_xlim([-50, 150])
    g.axes[1][6].set_xticks([1, 100])
    g.axes[1][6].set_xticklabels([1, 100])

    fig.tight_layout()
    if save:
        g.savefig(DAT_DIR + FILE + '_parameter.pdf', bbox_inches='tight')
    plt.show()

    # statistics
    df = df[['dev accuracy', 'test accuracy']].describe()
    if save:
        df.to_csv(DAT_DIR + FILE + '_stats.csv', sep='\t')
    print(df)


if __name__ == '__main__':
    df = pd.read_csv(DAT_DIR + FILE + '.csv', sep='\t', index_col=None)
    df.rename(columns={'dev_acc': 'dev', 'test_acc': 'test'}, inplace=True)

    plot(df, save=True)
