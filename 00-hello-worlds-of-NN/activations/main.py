import numpy as np

"""
Input Vector
"""

x = np.asarray([2, -3, 4, 1, 0, 5])

# ========================================================


"""
Sigmoid:

y = 1/(1+1/np.exp(x))

Goal: Get all the x values between 0 to 1

"""

sigmoid = 1/(1+1/np.exp(x))

# ========================================================


"""
Softmax:

y = np.exp(x)/np.sum(np.exp(x))

Goal: Get all the x values into a probability distribution

Use: Output for Categorical Cross Entropy. eg: [0, 0, 0, 0, 0, 1] for [2, -3, 4, 1, 0, 5]

"""

softmax = np.exp(x)/np.sum(np.exp(x))

# ========================================================

"""
ReLU:

y = np.maximum(0, x)

Goal: Helps solve the vanishing gradient

"""

relu = np.maximum(0, x)
