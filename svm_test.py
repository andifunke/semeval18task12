"""
in parts inspired by http://scikit-learn.org/stable/auto_examples/text/document_classification_20newsgroups.html#sphx-glr-auto-examples-text-document-classification-20newsgroups-py
"""
import re
from sklearn.metrics import accuracy_score
from os import listdir
from svm_train import *


def test_main():
    """ a given directory and/or filename overwrites the defaults """
    print('start testing')
    t0 = time()

    files = [f for f in listdir(OPTIONS['data_dir']) if re.match(OPTIONS['wv_file'], f)]

    clf = None
    options = None
    scaler = None
    for fname in files:
        if '_clf.' in fname:
            print('loading clf from', fname)
            clf = pd.read_pickle(path.join(OPTIONS['data_dir'], fname))
        elif '_scale.' in fname:
            print('loading scaler from', fname)
            scaler = pd.read_pickle(path.join(OPTIONS['data_dir'], fname))
        elif '_options.' in fname:
            print('loading options from', fname)

            options = json.load(open(path.join(OPTIONS['data_dir'], fname)))

    if clf is None:
        print("no clf file found. exit")
        return

    if options is None:
        print("no options file found. exit")
        return

    X, y_true = get_xy('dev')

    # using the scaler only if the data was trained on scaled values
    if scaler is not None:
        print('scaling values')
        X = scaler.transform(X)

    print('predict')
    y_pred = clf.predict(X)

    options['pred_acc'] = pred_acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    print('accuracy:', pred_acc)

    result_fname = path.join(OPTIONS['data_dir'], OPTIONS['wv_file'] + '_testresults.json')
    options['test_time'] = time() - t0

    print('writing results to', result_fname)
    print(options)
    with open(result_fname, 'w') as f:
        json.dump(options, f)

    print("done in {:f}s".format(options['test_time']))


if __name__ == '__main__':
    OPTIONS = get_options()
    test_main()
