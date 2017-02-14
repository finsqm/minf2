from kp_data_loader import KPDataLoader
from sklearn.cross_validation import KFold
import logging
import sys

logger = logging.getLogger()
TONIC = 0
TRAIN_TEST_RATIO = 0.9

def make_chords_per_note(XX, Y):
    YY = []
    for i, y in enumerate(Y):
        YY_i = []
        for j, frame in enumerate(XX[i]):
            for k, note in enumerate(frame):
                YY_i.append(y[j])
        YY.append(YY_i)
    return YY

def get_rid_of_XX_frames(XX):
    output = []
    for X in XX:
        output_i = []
        for x in X:
            output_i += x
        output.append(output_i)
    return output

def baseline_model_first_note(XX):
    full_guess = []
    for i, X in enumerate(XX):
        guess_i = []
        current_guess = None
        for j, x in enumerate(X):
            if j < 1:
                current_guess = x + 1
            guess_i.append(current_guess)
        full_guess.append(guess_i)
    return full_guess

def baseline_model_tonic(XX):
    full_guess = []
    for i, X in enumerate(XX):
        guess_i = []
        for j, x in enumerate(X):
            guess_i.append(TONIC)
        full_guess.append(guess_i)
    return full_guess

def get_accuracy(prediction, YY):
    total_count = 0
    num_correct = 0
    for i, Y in enumerate(YY):
        for j, y in enumerate(Y):
            total_count += 1
            guess = prediction[i][j]
            if guess == y:
                num_correct += 1
    return float(num_correct) / float(total_count)

logger.info("Getting Data ... ")
# Get data again just in case I messed up somewhere
loader = KPDataLoader()
for i in range(1,46):
    loader.load_file('ex{0}a.mid.csv'.format(i))
XX, Y = loader.get_XX_and_Y()
YY = make_chords_per_note(XX, Y)
XX = get_rid_of_XX_frames(XX)

# -----------------------------------------------

n = len(XX)
j = int(n - (float(n) * TRAIN_TEST_RATIO))

XX_train = XX[0:j]
YY_train = YY[0:j]

XX_test = XX[j:n]
YY_test = YY[j:n]

logger.info("First Note Baseline ...")
first_note_baseline_prediction = baseline_model_first_note(XX_test)
assert len(first_note_baseline_prediction) == len(YY_test)
first_note_baseline_accuracy = get_accuracy(first_note_baseline_prediction, YY_test)
logger.info("Accuracy: {0}".format(first_note_baseline_accuracy))

logger.info("Tonic Baseline ...")
tonic_baseline_prediction = baseline_model_tonic(XX_test)
assert len(tonic_baseline_prediction) == len(YY_test)
tonic_baseline_accuracy = get_accuracy(tonic_baseline_prediction, YY_test)
logger.info("Accuracy: {0}".format(tonic_baseline_accuracy))
# -----------------------------------------------
