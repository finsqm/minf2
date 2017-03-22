import keras
from keras.layers import LSTM
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Masking
from keras.layers import Dropout
from keras.layers.wrappers import TimeDistributed
from keras.layers import Embedding
import kp_data_loader as dl
from keras.preprocessing.sequence import pad_sequences
import os
from sklearn.model_selection import KFold

np.random.seed(42)

RESULTS_DIR_NAME = 'MINF_RESULTS_DIR'
results_folder = os.path.join(os.environ[RESULTS_DIR_NAME], 'kpcorpus-crossval')

loader = dl.KPDataLoader()
for i in range(1,46):
    loader.load_file('ex{0}a.mid.csv'.format(i))
X, Y, m = loader.get_XX_and_YY()

X_pad = pad_sequences(X, maxlen=m, padding='post')
Y_pad = pad_sequences(Y, maxlen=m, padding='post')

np.random.shuffle(X_pad)
np.random.shuffle(Y_pad)

sample_weights = np.ones((len(X), m))
for i in xrange(len(X)):
    for j in xrange(m):
        if (X_pad[i][j] == np.zeros(12)).all():
            sample_weights[i][j] = 0

# TODO: Implement cross val - work out if using held out testing set

for i in range(10): 
    model = Sequential()
    mask = np.zeros(12)
    model.add(Masking(mask_value=mask, input_shape=(m, 12)))
    model.add(LSTM(100, return_sequences=True, dropout_W=0.4, dropout_U=0.4))
    model.add(TimeDistributed(Dense(12, activation="softmax")))

    from keras.utils.np_utils import to_categorical

    model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              sample_weight_mode='temporal')

    n = len(X)
    j = int(n - (float(n) * 0.5))
    X_train, X_test = X_pad[:j, :], X_pad[j:, :]

    Y_train, Y_test = Y_pad[:j, :], Y_pad[j:, :]

    sample_weights_train, sample_weights_test = sample_weights[:j, :], sample_weights[j:, :]

    # # for custom metrics
    import numpy as K
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

    model.fit(X_train, Y_train, batch_size=j, nb_epoch=500, sample_weight=sample_weights_train)

    Y_prediction = model.predict_classes(X_test, batch_size=5)

    # Y_prediction = to_categorical(Y_prediction)
    # weighted_accuracy(Y_test, Y_prediction, sample_weights_test)
    acc = weighted_accuracy(Y_test, Y_prediction)

    filename = 'layers_{0}.txt'.format(i)
    data_path = os.path.join(results_folder, filename)

    with open(data_path, 'w') as f:
        f.write('{0}\n'.format(acc))
