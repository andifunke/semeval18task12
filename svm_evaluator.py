import random

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)


df = pd.read_csv('./svm/results.csv', sep='\t')
df = df[['dev_acc', 'test_acc']]

df = df.applymap(lambda x: x + random.uniform(-0.0005, 0.0005))
fig, ax = plt.subplots(ncols=3)

df.plot(ax=ax[0], x='dev_acc', y='test_acc', kind='scatter')
ax[0].set_xlim(.45, .65)
ax[0].set_ylim(.45, .65)
ax[0].set_xlabel('accuracy score: dev')
ax[0].set_ylabel('accuracy score: test')

print(df)
sns.distplot(df['dev_acc'], ax=ax[1], bins=5, rug=False, kde=True)
sns.distplot(df['test_acc'], ax=ax[1], bins=5, rug=False, kde=True)
df.plot(ax=ax[2], kind='box')
plt.show()
