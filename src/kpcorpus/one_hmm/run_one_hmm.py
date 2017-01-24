from kp_hmm import HMM
from kp_data_loader import KPDataLoader, sequence
from sklearn.cross_validation import KFold
import logging
import sys
from seqlearn import hmm, perceptron

logger = logging.getLogger()
logging.basicConfig(filename='../../../results/run_one_hmm.log',level=logging.DEBUG)

# -----------------------------------------------

# loader = KPDataLoader()
# for i in range(1,46):
#     loader.load_file('ex{0}a.mid.csv'.format(i))
# XX, Y = loader.get_XX_and_YY()
#
# # -----------------------------------------------
#
# n = len(XX)
# j = int(n - (float(n) * 0.1))
#
# XX_train = XX[0:j]
# Y_train = Y[0:j]
#
# XX_test = XX[j:n]
# Y_test = Y[j:n]
#
# # -----------------------------------------------
#
# model = HMM()
# model.train(XX_train, Y_train)
# accuracy, y_pred = model.test(XX_test, Y_test)
#
# logger.info("Accuracy: {0}".format(accuracy))
#
# def cross_val(n=10):
# 		"""
# 		n : n-crossvalidation
# 		"""
#
# 		L = len(XX_train)
# 		kf = KFold(L,n_folds=n)
#
# 		models = []
# 		scores = []
#
# 		c = 0
#
# 		for c, (train_indexes, val_indexes) in enumerate(kf):
#
# 			logger.debug("On Fold " + str(c))
#
# 			xx_train = []
# 			y_train = []
# 			xx_val = []
# 			y_val = []
# 			for i in train_indexes:
# 				xx_train.append(XX_train[i][:])
# 				y_train.append(Y_train[i][:])
# 			for j in val_indexes:
# 				xx_val.append(XX_train[j][:])
# 				y_val.append(Y_train[j][:])
#
# 			model = HMM()
#
# 			logger.debug(str(len(xx_train)) + "," + str(len(y_train)))
# 			model.train(xx_train,y_train)
#
# 			logger.debug("Testing ...")
# 			score, _ = model.test(xx_val,y_val)
#
# 			logger.debug("Fold " + str(c) + " scored " + str(score))
#
# 			models.append(model)
# 			scores.append(score)
#
# 		max_score = max(scores)
#
# 		print max_score
#
# 		max_index = 0
# 		for idx, score in enumerate(scores):
# 			if score == max_score:
# 				max_index = idx
# 				break
#
# 		logger.info("Final Test ...")
#
# 		score, _ = models[max_index].test(XX_test,Y_test)
# 		logger.info("Final Accuracy: {0}".format(score))
#
# 		return models[max_index]
#
# model = cross_val(10)

# -----------------------------------------------

# Now with seqlearn

logger.info("Seqlearn time ... ")
# Get data again just in case I messed up somewhere
loader = KPDataLoader()
for i in range(1,46):
    loader.load_file('ex{0}a.mid.csv'.format(i))
XX, YY = loader.get_XX_and_YY()

# -----------------------------------------------

n = len(XX)
j = int(n - (float(n) * 0.1))

XX_train = XX[0:j]
YY_train = YY[0:j]

X_train, Y_train, L_train = sequence(XX_train, YY_train)

XX_test = XX[j:n]
YY_test = YY[j:n]

X_test, Y_test, L_test = sequence(XX_test, YY_test)

logger.info('Building HMM model ...')
model = hmm.MultinomialHMM()
logger.info('Training model ...')
model.fit(X_train, Y_train, L_train)
logger.info('Testing model ...')
accuracy = model.score(X_test, Y_test, L_test)

logger.info("Accuracy: {0}".format(accuracy))
# for i in model.predict(X_test, L_test):
#     print "{0} ".format(i),

logger.info('Building Percpetron model ...')
model = perceptron.StructuredPerceptron()
logger.info('Training model ...')
model.fit(X_train, Y_train, L_train)
logger.info('Testing model ...')
accuracy = model.score(X_test, Y_test, L_test)
logger.info("Accuracy: {0}".format(accuracy))
# -----------------------------------------------
