from keras.layers import LSTM
import numpy as np
import cPickle
from keras.models import Sequential
import data_loader as dl
from keras.layers import Dense
from keras.layers import Masking
from keras.layers.wrappers import TimeDistributed
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import KFold, LeaveOneOut
import logging

np.random.seed(42)

data = dl.DataLoader()
X, Y, m = data.load()

X_pad = pad_sequences(X, maxlen=m, padding='post')
Y_pad = pad_sequences(Y, maxlen=m, padding='post')
perm = np.random.permutation(len(X_pad))
X_pad = X_pad[perm]
Y_pad = Y_pad[perm]

sample_weights = np.ones((273, m))
for i in xrange(273):
    for j in xrange(m):
        if (X_pad[i][j] == np.zeros(12)).all():
            sample_weights[i][j] = 0

def weighted_accuracy(y_true, y_pred):
    # Only for testing
    #     score_array = K.equal(K.argmax(y_true, axis=-1),
    #                           K.argmax(y_pred, axis=-1))
    #     score_array *= weights
    #     score_array /= K.mean(K.cast(K.not_equal(weights, 0), K.floatx()))
    #     return K.mean(score_array)
    total = 0.0
    count = 0.0
    for i, y_i in enumerate(y_true):
        for j, y_ij in enumerate(y_i):
            if sum(y_ij) > 0:
                total += y_ij[y_pred[i][j]]
                count += 1
    return total / count

def cross_val(X_pad, Y_pad, sample_weights, m, n=10):
    """
    n : n-crossvalidation
    """

    L = len(X_pad)
    kf = KFold(n_splits=n)

    models = []
    scores = []
    hists = []

    c = 0

    for c, (train_indexes, val_indexes) in enumerate(kf.split(X_pad)):

        print "On Fold " + str(c)

        xx_train = []
        y_train = []
        xx_val = []
        y_val = []
        sample_weights_train = []
        sample_weights_test = []

        for i in train_indexes:
            xx_train.append(X_pad[i][:])
            y_train.append(Y_pad[i][:])
            sample_weights_train.append(sample_weights[i][:])
        for j in val_indexes:
            xx_val.append(X_pad[j][:])
            y_val.append(Y_pad[j][:])
            sample_weights_test.append(sample_weights[j][:])

        l = len(xx_train)

        xx_train = np.asarray(xx_train)
        xx_val = np.asarray(xx_val)
        y_train = np.asarray(y_train)
        y_val = np.asarray(y_val)
        sample_weights_train = np.asarray(sample_weights_train)
        sample_weights_test = np.asarray(sample_weights_test)

        model = Sequential()

        mask = np.zeros(12)
        model.add(Masking(mask_value=mask, input_shape=(m, 12)))
        model.add(LSTM(50, return_sequences=True, dropout_W=0.4, dropout_U=0.4))
        model.add(TimeDistributed(Dense(12, activation="softmax")))

        model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              sample_weight_mode='temporal')

        hist = model.fit(xx_train, y_train, batch_size=l, nb_epoch=50, sample_weight=sample_weights_train)
        hists.append(hist)

        print "Testing ..."
        Y_prediction = model.predict_classes(xx_val, batch_size=5)
        accuracy = weighted_accuracy(y_val, Y_prediction)

        print "Fold " + str(c) + " scored " + str(accuracy)

        models.append(model)
        scores.append(accuracy)

    return scores, hists, models

scores, hists, models = cross_val(X_pad, Y_pad, sample_weights, m, n=10)
mean = np.mean(scores)
std = np.std(scores)

with open('LSTM-cross-val-script.txt', 'w') as f:
    f.write('Mean: {0}\n'.format(mean))
    f.write('Std: {0}\n'.format(std))

dump = dict()
dump['models'] = models
dump['hists'] = hists

with opne('LSTM-cross-val-hists-models.pkl', 'w') as f:
    cPickle.dump(dump, f)
