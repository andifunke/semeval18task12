""" compare the accuracy distribution of dev and test validation from the SVM models """
import random
from os import listdir

import pandas as pd
import matplotlib.pyplot as plt
import re
import seaborn as sns

from results_evaluator import tprint

sns.set(color_codes=True)

dat_dir = '../out/'
files = [f for f in listdir(dat_dir) if re.match(r'^svm_results.*\.csv$', f)]
colors = ['b', 'g', 'r', 'y', 'c', 'm', 'k', 'w']
acc_col = ['dev_acc', 'test_acc']

results = []
fig, ax = plt.subplots(ncols=3)
for i, fname in enumerate(files):
    d = pd.read_csv(dat_dir + fname, sep='\t', index_col=0)
    results.append(d)
    d[acc_col] = d[acc_col].applymap(lambda x: x + random.uniform(-0.0005, 0.0005))
    d.plot(ax=ax[0], x='dev_acc', y='test_acc', kind='scatter', c=colors[i])

df = pd.concat(results)

ax[0].set_xlim(.45, .65)
ax[0].set_ylim(.45, .65)
ax[0].set_xlabel('accuracy score: dev')
ax[0].set_ylabel('accuracy score: test')

tprint(df)
sns.distplot(df['dev_acc'], ax=ax[1], bins=5, rug=False, kde=True)
sns.distplot(df['test_acc'], ax=ax[1], bins=5, rug=False, kde=True)
df.boxplot(acc_col, ax=ax[2])
plt.show()
