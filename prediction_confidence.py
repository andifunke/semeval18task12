import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

np.set_printoptions(linewidth=8000)

ser = pd.read_csv('./data/dev/dev-only-labels.txt', sep='\t', index_col=0, header=0)
y_true = ser.values.flatten()

y_svm = [
    './svm/custom_embedding_wv2_sg_ns_iter20_50.predictions.npy',
]
y_nn = [
    './out/model_2018-01-10_21-21-59-946390_rn01_ep07_ac0.636.probabilities.npy',
    './out/model_2018-01-10_21-21-59-946390_rn02_ep08_ac0.680.probabilities.npy',
    './out/model_2018-01-10_21-21-59-946390_rn03_ep08_ac0.646.probabilities.npy',
]

y_svm_probabilities = np.asarray([np.load(y)[:, :1].flatten() for y in y_svm])
y_nn_probabilities = np.asarray([np.load(y).flatten() for y in y_nn])
length = len(y_nn_probabilities[0])

y_svm_predictions = (y_svm_probabilities >= 0.5)
y_nn_predictions = (y_nn_probabilities >= 0.5)

for y_pred in y_svm_predictions:
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    print('accuracy: {:.3f}'.format(acc))
for y_pred in y_nn_predictions:
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    print('accuracy: {:.3f}'.format(acc))


""" confidence vote """

def get_conf(prob):
    return np.abs(prob - 0.5)


prob = np.full(length, 0.5, dtype=np.float64)

for prob_tmp in np.concatenate((y_svm_probabilities, y_nn_probabilities)):
    conf = get_conf(prob)
    conf_tmp = get_conf(prob_tmp)
    vote_tmp = conf_tmp > conf
    prob[vote_tmp] = prob_tmp[vote_tmp]
    pred = prob >= 0.5
    acc = accuracy_score(y_true=y_true, y_pred=pred)

print('combined accuracy by confidence: {:.3f}'.format(acc))


""" majority vote """

pred_conf = pred
all_predictions = np.concatenate((y_svm_predictions, y_nn_predictions))

pred = np.zeros(length)

pred = np.mean(all_predictions, axis=0)
# in case of a tie use the predictions from the condifdence vote
tie = np.isclose(pred, 0.5)
pred[tie] = pred_conf[tie]
pred = (pred >= 0.5)
acc = accuracy_score(y_true=y_true, y_pred=pred)
print('combined accuracy by majority vote: {:.3f}'.format(acc))
