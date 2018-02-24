""" simple script to compare dev and test accuracy of answer files in dicretory d """
from os import listdir
import pandas as pd
import matplotlib.pyplot as plt
import re
from sklearn.metrics import accuracy_score
from constants import *


dev_true = pd.read_csv(FILES['dev'], sep='\t', index_col=ID, usecols=[0, 3], names=[ID, 'true'],
                       header=0)
test_true = pd.read_csv(FILES['test'], sep='\t', index_col=ID, usecols=[0, 3], names=[ID, 'true'],
                        header=0)

d = '../results/new/'

dev_answers = [f for f in listdir(d) if re.match(r'^answer-dev.*', f)]
test_answers = [f for f in listdir(d) if re.match(r'^answer-tst.*', f)]


def get_answers(files, df_true):
    accs = dict()
    for f in files:
        idx = f[10:13]
        answer = pd.read_csv(d + f, sep='\t', index_col='#id', names=['#id', 'pred'], header=0)
        df = df_true.join(answer)
        acc = accuracy_score(y_true=df['true'].values, y_pred=df['pred'].values)
        accs[idx] = acc
    return sorted(accs.items())


accs_dev = get_answers(dev_answers, dev_true)
accs_test = get_answers(test_answers, test_true)

accs_dev = [x[1] for x in accs_dev]
accs_test = [x[1] for x in accs_test]

df = pd.DataFrame({'dev': accs_dev, 'test': accs_test})
print(df)
df = df.describe()
print(df)

plt.scatter(x=accs_dev, y=accs_test)
plt.xlabel('accuracy score: dev')
plt.ylabel('accuracy score: test')
plt.show()

