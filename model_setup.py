import math
import numpy as np


# Nonlinear activation function that converts neuron outputs.
# Necessary to introduce nonlinearity to neural network
# (otherwise the result of the network is basically the output
# of a giant linear function which is pretty useless for
# nonlinear problems).
def activation(input):
    return 1 / (1 + math.e ** -input)