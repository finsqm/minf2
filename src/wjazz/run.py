from data_loader import DataLoader
import math
import numpy as np
from seqlearn import hmm, perceptron
import logging
import sys

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)

TRAINING_RATIO = 0.9


logger.info('Getting data ...')
dl = DataLoader()
X, Y, L = dl.load()
logger.info('Data Loaded.')

# Number of songs
N = len(L)
l_split_point = int(math.ceil(N * TRAINING_RATIO))
xy_split_point = np.sum(L[:(l_split_point)])

X_train = X[:xy_split_point]
Y_train = Y[:xy_split_point]
L_train = L[:l_split_point]

len_train = np.sum(L_train)
len_X_train = len(X_train)
len_Y_train = len(Y_train)
assert len_train == len_X_train, "sum of training lengths: {0} not equal to len(X_train): {1}".format(len_train, len_X_train)
assert len_train == len_Y_train, "sum of training lengths: {0} not equal to len(Y_train): {1}".format(len_train, len_Y_train)

X_test = X[xy_split_point:]
Y_test = Y[xy_split_point:]
L_test = L[l_split_point:]

len_test = np.sum(L_test)
len_X_test = len(X_test)
len_Y_test = len(Y_test)
assert len_test == len_X_test, "sum of testing lengths: {0} not equal to len(X_test): {1}".format(len_test, len_X_test)
assert len_test == len_Y_test, "sum of testing lengths: {0} not equal to len(Y_test): {1}".format(len_test, len_Y_test)

logger.info('Building HMM model ...')
model = hmm.MultinomialHMM()
logger.info('Training model ...')
model.fit(X_train, Y_train, L_train)
logger.info('Testing model ...')
accuracy = model.score(X_test, Y_test, L_test)
logger.info("Accuracy: {0}".format(accuracy))
# for i in model.predict(X_test):
#     print "{0} ".format(i),

logger.info('Building model ...')
model = perceptron.StructuredPerceptron()
logger.info('Training model ...')
model.fit(X_train, Y_train, L_train)
logger.info('Testing model ...')
accuracy = model.score(X_test, Y_test, L_test)
logger.info("Accuracy: {0}".format(accuracy))
# for i in model.predict(X_test):
#     print "{0} ".format(i),
