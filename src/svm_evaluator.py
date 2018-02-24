""" compare the accuracy distribution of dev and test validation from the SVM models """
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dat_dir = '../out/'
fname = 'svm_results_alt-split_odd-ratio'
colors = ['navy', 'darkmagenta', 'r', 'm', 'b', 'c', 'y', 'g', 'k', 'w']
acc_col = ['dev', 'test']

df = pd.read_csv(dat_dir + fname + '.csv', sep='\t', index_col=None)
df.rename(columns={'dev_acc': 'dev', 'test_acc': 'test'}, inplace=True)

# this takes the data points a little bit off the grid and enhances the plots
add_jitter = False
if add_jitter:
    df[acc_col] = df[acc_col].applymap(lambda x: x + random.uniform(-0.00025, 0.00025))

no_scale = df.loc[~df.scale]
scale = df.loc[df.scale]

sns.set(color_codes=True, font_scale=0.8)
sns.set_style("whitegrid", {'legend.frameon': True})

fig2, ax2 = plt.subplots(figsize=(5, 5))
ax2.set_xlim(.475, .675)
ax2.set_ylim(.475, .675)
ax2.set_xticks([0.475, 0.5, 0.525, 0.55, 0.575, 0.6, 0.625, 0.65, 0.675])
ax2.set_yticks([0.475, 0.5, 0.525, 0.55, 0.575, 0.6, 0.625, 0.65, 0.675])
sns.regplot(x=no_scale['dev'], y=no_scale['test'], ax=ax2, scatter=True, fit_reg=False, color=colors[0],
            label='original embeddings')
sns.regplot(x=scale['dev'], y=scale['test'], ax=ax2, scatter=True, fit_reg=False, color=colors[1],
            label='scaled embeddings')
ax2.set_xlabel('accuracy score: dev', weight='bold', size=11)
ax2.set_ylabel('accuracy score: test', weight='bold', size=11)
ax2.legend()
fig2.tight_layout()
fig2.savefig(dat_dir + fname + '_scatter1.pdf', bbox_inches='tight')
plt.show()

fig2, ax2 = plt.subplots(figsize=(5, 5))
ax2.set_xlim(.475, .675)
ax2.set_ylim(.475, .675)
ax2.set_xticks([0.475, 0.5, 0.525, 0.55, 0.575, 0.6, 0.625, 0.65, 0.675])
ax2.set_yticks([0.475, 0.5, 0.525, 0.55, 0.575, 0.6, 0.625, 0.65, 0.675])
sns.regplot(x=no_scale['dev'], y=no_scale['test'], ax=ax2, scatter=True, fit_reg=False, color=colors[0],
            label='original embeddings')
sns.regplot(x=scale['dev'], y=scale['test'], ax=ax2, scatter=True, fit_reg=False, color=colors[0],
            label='scaled embeddings')
ax2.set_xlabel('accuracy score: dev', weight='bold', size=11)
ax2.set_ylabel('accuracy score: test', weight='bold', size=11)
fig2.tight_layout()
fig2.savefig(dat_dir + fname + '_scatter2.pdf', bbox_inches='tight')
plt.show()


fig, ax = plt.subplots(ncols=3, figsize=(15, 5))

sns.regplot(x=no_scale['dev'], y=no_scale['test'], ax=ax[0], scatter=True, fit_reg=False, color=colors[0],
            label='original embeddings')
sns.regplot(x=scale['dev'], y=scale['test'], ax=ax[0], scatter=True, fit_reg=False, color=colors[1],
            label='scaled embeddings')
ax[0].set_xlim(.45, .675)
ax[0].set_ylim(.45, .675)
ax[0].set_xlabel('accuracy score: dev', weight='bold')
ax[0].set_ylabel('accuracy score: test', weight='bold')
ax[0].legend(loc='lower right')

sns.distplot(df['dev'], ax=ax[1], bins=5, rug=False, kde=True, label='dev')
sns.distplot(df['test'], ax=ax[1], bins=5, rug=False, kde=True, label='test')
ax[1].set_xlim(.45, .675)
ax[1].set_xlabel('distribution of accuracy scores', weight='bold')
ax[1].legend()

sns.boxplot(data=df[acc_col], ax=ax[2], notch=True, width=.3)
ax[2].set_xlabel('distribution of accuracy scores', weight='bold')
ax[2].set_ylim(.45, .675)
fig.tight_layout()
fig.savefig(dat_dir + fname + '.pdf', bbox_inches='tight')
plt.show()

df.rename(columns={'dev': 'dev accuracy', 'test': 'test accuracy', 'dims': 'dimensions', 'scale': 'scaled'}, inplace=True)
g = sns.pairplot(df, y_vars=['dev accuracy', 'test accuracy'],
                 x_vars=['dev accuracy', 'test accuracy', 'embedding', 'dimensions', 'kernel', 'scaled', 'C'])
g.axes[0][0].set_ylim(0.45, 0.675)
g.axes[1][0].set_ylim(0.45, 0.675)
g.axes[1][0].set_xlim(0.45, 0.675)
g.axes[1][1].set_xlim(0.45, 0.675)
g.axes[1][0].set_xticks([0.45, 0.5, 0.55, 0.6, 0.65])
g.axes[1][1].set_xticks([0.45, 0.5, 0.55, 0.6, 0.65])
# embedding
g.axes[1][2].set_xticklabels([1, 2, 3, 4, 5, 6])
# dimensions
g.axes[1][3].set(xscale='log')
g.axes[1][3].set_xticks([25, 50, 100, 300])
g.axes[1][3].set_xticklabels([25, 50, 100, 300])
# scaled
g.axes[1][5].set_xticks([0, 1])
g.axes[1][5].set_xticklabels(['False', 'True'])
# C
g.axes[1][6].set_xticklabels(['', 1, 50, 100])
plt.show()
g.savefig(dat_dir + fname + '_parameter.pdf', bbox_inches='tight')

df = df[['dev accuracy', 'test accuracy']].describe()
df.to_csv(dat_dir + fname + '_stats.csv', sep='\t')
print(df)
