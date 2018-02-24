""" compare the accuracy distribution of dev and test validation from the SVM models """
import random
from os import listdir

import pandas as pd
import matplotlib.pyplot as plt
import re
import seaborn as sns

from results_evaluator import tprint

sns.set(color_codes=True)
sns.set_style("whitegrid", {'legend.frameon':True})

dat_dir = '../out/'
files = sorted([f for f in listdir(dat_dir) if re.match(r'^svm_results.*\.csv$', f)])
colors = ['navy', 'darkmagenta', 'r', 'm', 'b', 'c', 'y', 'g', 'k', 'w']
acc_col = ['dev', 'test']
label = {'svm_results_alt_split_fair_swap.csv': 'fair split ratio',
         'svm_results_swap.csv': 'original split ratio',
         'svm_results_alt_split_fair_swap_scale.csv': 'fair split ratio (scaled)'
         }

results = []

fig_size = (15, 5)
fig, ax = plt.subplots(ncols=3, figsize=fig_size)

for i, fname in enumerate(files):
    d = pd.read_csv(dat_dir + fname, sep='\t', index_col=0)
    d.rename(columns={'dev_acc': 'dev', 'test_acc': 'test'}, inplace=True)
    results.append(d)
    d[acc_col] = d[acc_col].applymap(lambda x: x + random.uniform(-0.0001, 0.0001))
    sns.regplot(x=d['dev'], y=d['test'], ax=ax[0], scatter=True, fit_reg=False, color=colors[i], label=label[fname])
ax[0].set_xlim(.475, .675)
ax[0].set_ylim(.475, .675)
ax[0].set_xlabel('accuracy score: dev', weight='bold')
ax[0].set_ylabel('accuracy score: test', weight='bold')
ax[0].legend(loc='upper left')

df = pd.concat(results)

sns.distplot(df['dev'], ax=ax[1], bins=5, rug=False, kde=True, label='dev')
sns.distplot(df['test'], ax=ax[1], bins=5, rug=False, kde=True, label='test')
ax[1].set_xlabel('distribution of accuracy scores', weight='bold')
ax[1].legend()

sns.boxplot(data=df[acc_col], ax=ax[2], notch=True, width=.3)
ax[2].set_xlabel('distribution of accuracy scores', weight='bold')
fig.tight_layout()
plt.show()

df.rename(columns={'dev': 'dev accuracy', 'test': 'test accuracy', 'dims': 'dimensions', 'scale': 'scaled'}, inplace=True)
g = sns.pairplot(df, y_vars=['dev accuracy', 'test accuracy'],
                 x_vars=['dev accuracy', 'test accuracy', 'embedding', 'dimensions', 'kernel', 'scaled', 'C'])
g.axes[0][0].set_ylim(0.45, 0.675)
g.axes[1][0].set_ylim(0.45, 0.675)
# embedding
g.axes[1][2].set_xticklabels([1, 2, 3, 4, 5, 6])
# dimensions
g.axes[1][3].set(xscale="log")
g.axes[1][3].set_xticks([25, 50, 100, 300])
g.axes[1][3].set_xticklabels([25, 50, 100, 300])
# scaled
g.axes[1][5].set_xticks([0, 1])
g.axes[1][5].set_xticklabels(['False', 'True'])
# C
g.axes[1][6].set_xticklabels(['', 1, 50, 100])
plt.show()
