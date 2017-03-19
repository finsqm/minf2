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

TRAINING_RATIO = 0.5
TONIC = 0

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

def guess_tonic(Y):
    guess = []
    for y in Y:
        guess.append(TONIC)
    return guess

def guess_first_note(X, Y, L):
    song_position = 0
    chord_guess = None
    prediction = []
    for l in L:
        for i in range(l):
            index = song_position + i
            if i == 0:
                chord_guess = X[index]
            prediction.append(chord_guess)
        song_position += l
    return prediction



def get_accuracy(prediction, Y):
    total_count = 0
    num_correct = 0
    for i, y in enumerate(Y):
        total_count += 1
        guess = prediction[i]
        if guess == y:
            num_correct += 1
    return float(num_correct) / float(total_count)

logger.info("Tonic Prediction ...")
tonic_baseline_prediction = guess_tonic(Y_test)
assert len(tonic_baseline_prediction) == len(Y_test)
tonic_baseline_accuracy = get_accuracy(tonic_baseline_prediction, Y_test)
logger.info("Accuracy: {0}".format(tonic_baseline_accuracy))

logger.info("First Note Prediction ...")
first_note_baseline_prediction = guess_first_note(X_test, Y_test, L_test)
assert len(first_note_baseline_prediction) == len(Y_test)
first_note_baseline_accuracy = get_accuracy(first_note_baseline_prediction, Y_test)
logger.info("Accuracy: {0}".format(first_note_baseline_accuracy))

# -----------------------------------------------
# logger = logging.getLogger()
# TONIC = 1
# TRAIN_TEST_RATIO = 0.1
#
# def make_chords_per_note(XX, Y):
#     YY = []
#     for i, y in enumerate(Y):
#         YY_i = []
#         for j, frame in enumerate(XX[i]):
#             for k, note in enumerate(frame):
#                 YY_i.append(y[j])
#         YY.append(YY_i)
#     return YY
#
# def get_rid_of_XX_frames(XX):
#     output = []
#     for X in XX:
#         output_i = []
#         for x in X:
#             output_i += x
#         output.append(output_i)
#     return output
#
# def baseline_model_first_note(XX):
#     full_guess = []
#     for i, X in enumerate(XX):
#         guess_i = []
#         current_guess = None
#         for j, x in enumerate(X):
#             if j < 1:
#                 current_guess = x + 1
#             guess_i.append(current_guess)
#         full_guess.append(guess_i)
#     return full_guess
#
# def baseline_model_tonic(XX):
#     full_guess = []
#     for i, X in enumerate(XX):
#         guess_i = []
#         for j, x in enumerate(X):
#             guess_i.append(TONIC)
#         full_guess.append(guess_i)
#     return full_guess
#
# def get_accuracy(prediction, YY):
#     total_count = 0
#     num_correct = 0
#     for i, Y in enumerate(YY):
#         for j, y in enumerate(Y):
#             total_count += 1
#             guess = prediction[i][j]
#             if guess == y:
#                 num_correct += 1
#     return float(num_correct) / float(total_count)
# logger.info("First Note Baseline ...")
# first_note_baseline_prediction = baseline_model_first_note(XX_test)
# assert len(first_note_baseline_prediction) == len(YY_test)
# first_note_baseline_accuracy = get_accuracy(first_note_baseline_prediction, YY_test)
# logger.info("Accuracy: {0}".format(first_note_baseline_accuracy))
#
# logger.info("Tonic Baseline ...")
# tonic_baseline_prediction = baseline_model_tonic(XX_test)
# assert len(tonic_baseline_prediction) == len(YY_test)
# tonic_baseline_accuracy = get_accuracy(tonic_baseline_prediction, YY_test)
# logger.info("Accuracy: {0}".format(tonic_baseline_accuracy))
