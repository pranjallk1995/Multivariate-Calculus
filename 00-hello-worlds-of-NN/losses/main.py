import numpy as np

"""
Binary Cross Entropy

L = -actual*np.log(predicted)-(1-actual)*np.log(1-predicted)

"""

actual = 1
predicted = 0.6

LB = -actual*np.log(predicted)-(1-actual)*np.log(1-predicted)

"""
Categorical Cross Entropy

L = -np.sum(actual*np.log(predicted))

"""

actual = np.asarray([0, 0, 1])
predicted = np.asarray([0.2, 0.25, 0.55])

LC = -np.sum(actual*np.log(predicted))

"""
Mean Square Error

L = np.square(actual - predicted)

"""

actual = np.asarray([0, 0, 1])
predicted = np.asarray([0.2, 0.25, 0.55])

LM = np.sum(np.square(actual - predicted)) * (1/len(actual))
