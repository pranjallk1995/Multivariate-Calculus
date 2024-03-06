"""
Gradient Descent

weights = weights - learning_rate * d(Loss function)

Goal: To gradually reach the local minima in the Loss function.

Note: Uses the entire dataset in the d(Loss function)

"""

# ========================================================

"""
Stochastic Gradient Descent

weights = weights - learning_rate * d(Loss function at point i) | for all the datapoints i

Goal: To reach the local minima in the loss function much faster as it starts updating the weights one point or a small batch at a time.

"""

# ========================================================

"""
Adagrad (Adaptive Gradient Descent)

weights_i = weights_i - adaptive_learning_rate * d_weight_i(Loss function) for all weights i

where:
    adaptive_learning_rate = global_learning_rate/np.sqrt(np.sum(np.square(d(Loss function upto current iteration))) + smoothing factor)

    the smoothing factor helps with division by zero cases, and upto current iteration makes sure that the path taken upto the current point is known.

Goal: adapts the learning rate individially for each weight_i. This is to make sure it gives large learning rate for weight_i if it is infrequent and vise versa.
        This will be true because d(Loss function) will be large.

Note: Useful for spase data, but that means d(Loss function) can become very large and the final adaptive learning rate is very small.

"""

# ========================================================

"""
Adadelta

Basically, handles the issue of very large d_weight_i(Loss function) by taking weighted average of change in step size 
of the weight taken in that direction. i.e. (weighted avg of change in step size along the weight)/(weighted avg of d_weight_i(Loss function))

The numerator also increases as d_weight_i(Loss function) increases.

"""
