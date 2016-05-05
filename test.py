import math
import numpy

np = numpy

# Define input vector
x = np.array([0.05, 0.10])

# Define weight matrix for input-hidden1
w1 = np.array([0.15, 0.25])
w2 = np.array([0.20, 0.30])
W = np.matrix([w1, w2])
bias = 0

# HIDDEN LAYER 1
# Activations
h = W.dot(x) + bias
# Activation functions
