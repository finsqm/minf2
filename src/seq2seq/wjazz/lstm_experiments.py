#!/usr/bin/env python

from data_loader import DataLoader
from keras.preprocessing.sequence import pad_sequences
import numpy as np

from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Masking
from keras.layers.wrappers import TimeDistributed


data = DataLoader()
X, Y, m = data.load()

X_pad = pad_sequences(X, maxlen=m, padding='post')
Y_pad = pad_sequences(Y, maxlen=m, padding='post')

batch_size = X_pad.shape[0]
dim = X_pad.shape[2]

sample_weights = np.ones((batch_size, m))
for i in xrange(batch_size):
    for j in xrange(m):
        if (X_pad[i][j] == np.zeros(12)).all():
            sample_weights[i][j] = 0
