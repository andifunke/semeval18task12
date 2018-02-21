from os import listdir
import pandas as pd
from constants import FILES
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)

true = pd.read_csv(FILES['test_true'], sep='\s+', index_col=[0], header=None, comment='#', squeeze=True)
d = './data/submissions/'
directory = listdir(d)
files = [f for f in directory]
answers = {f[:-4]: pd.read_csv(d + f, sep='\s+', index_col=[0], header=None, comment='#', squeeze=True) for f in files}
df = pd.DataFrame(answers)
df = df.eq(true, axis=0)
sums = df.sum(axis=1)
value_counts = sums.value_counts()
value_counts = value_counts.sort_index(ascending=False)
# value_counts /= value_counts.sum()
fig, ax = plt.subplots()
ax2 = ax.twinx()
ax.set_xlabel("correctly answered by [x] teams")
ax.set_ylabel("number of labels")
ax2.set_ylabel("percent of labels")
ymin, ymax = ax.get_ylim()
norm = lambda x: x * 100 / value_counts.sum()
ax2.set_ylim((norm(ymin), norm(ymax)))
ax2.plot([], [])
value_counts.plot(ax=ax, kind="bar")
plt.show()
