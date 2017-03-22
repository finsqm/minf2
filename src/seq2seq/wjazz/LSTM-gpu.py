from keras.layers import LSTM
import numpy as np
import cPickle
import os
from keras.models import Sequential
import data_loader as dl
from keras.layers import Dense
from keras.layers import Masking
from keras.layers.wrappers import TimeDistributed
from keras.preprocessing.sequence import pad_sequences

results_dir = os.environ.get('MINF_RESULTS_DIR')

np.random.seed(42)

data = dl.DataLoader()

X, Y, m = data.load()

X_pad = pad_sequences(X, maxlen=m, padding='post')
Y_pad = pad_sequences(Y, maxlen=m, padding='post')

np.random.shuffle(X_pad)
np.random.shuffle(Y_pad)

sample_weights = np.ones((273, m))
for i in xrange(273):
    for j in xrange(m):
        if (X_pad[i][j] == np.zeros(12)).all():
            sample_weights[i][j] = 0

model = Sequential()
accuracies = dict()
# Just do once, can't be bothered removing loop
for i in [1]:
    mask = np.zeros(12)
    model.add(Masking(mask_value=mask, input_shape=(m, 12)))
    model.add(LSTM(50, return_sequences=True, dropout_W=0.4, dropout_U=0.4))
    model.add(TimeDistributed(Dense(12, activation="softmax")))
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  sample_weight_mode='temporal')
    X_train, X_test = X_pad[:136, :], X_pad[136:, :]
    Y_train, Y_test = Y_pad[:136, :], Y_pad[136:, :]
    sample_weights_train, sample_weights_test = sample_weights[:136, :], sample_weights[136:, :]
    # # for custom metrics
    # def weighted_accuracy(y_true, y_pred):
    #     score_array *= sample_weights
    # score_array /= K.mean(K.cast(K.not_equal(weights, 0), K.floatx()))

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
    history = model.fit(X_train, Y_train, batch_size=50, nb_epoch=500, sample_weight=sample_weights_train)
    Y_prediction = model.predict_classes(X_test, batch_size=10)
    acc = weighted_accuracy(Y_test, Y_prediction), history.history

    with open(os.path.join(results_dir, 'LSTM-shuffle-gpu.pkl'), 'w') as f:
        cPickle.dump(acc, f)
