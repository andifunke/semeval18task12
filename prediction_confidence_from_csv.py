import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from os import listdir
import re
import matplotlib.pyplot as plt

from results_evaluator import tprint

np.set_printoptions(linewidth=8000)


def get_conf(prob):
    return np.abs(prob - 0.5)


def plot(df, name, show=True):
    fig = plt.figure()
    df.plot()
    plt.xlabel('number of models')
    plt.ylabel('accuracy')
    plt.title('Ensemble scores')
    if show:
        plt.show()
    fig.savefig('./figures/' + name + '.pdf')


def main():
        k = 'tensorL05con2redo2'
        df = pd.read_csv('/media/andreas/Linux_Data/hpc-semeval/tensorL05con2redo2_best.csv', sep='\t', index_col=0)
        tprint(df)
        plot(df, k + 'all_')


if __name__ == '__main__':
    main()
