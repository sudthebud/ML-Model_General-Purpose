###########
# IMPORTS #
###########
from enum import Enum

import numpy as np


####################
# CONSTS AND ENUMS #
####################
leakyReLU_Alpha = 0.01

class ActivationFunc(Enum):
    SIGMOID = 0
    TANH = 1
    RELU = 2
    LEAKY_RELU = 3


#############
# FUNCTIONS #
#############

# Nonlinear activation function that converts neuron outputs.
# Necessary to introduce nonlinearity to neural network
# (otherwise the result of the network is basically the output
# of a giant linear function which is pretty useless for
# nonlinear problems).
def activation(matrix, activationFunc):
    if activationFunc == ActivationFunc.SIGMOID: return 1 / (1 + np.exp(-matrix))
    elif activationFunc == ActivationFunc.TANH: return (np.exp(matrix) - np.exp(-matrix)) / (np.exp(matrix) + np.exp(-matrix))
    elif activationFunc == ActivationFunc.RELU: return np.where(matrix >= 0, matrix, 0)
    elif activationFunc == ActivationFunc.LEAKY_RELU: return np.where(matrix >= 0, matrix, -leakyReLU_Alpha * matrix)

    raise ValueError("Invalid activation function")

# Create weights and biases for each hidden layer + output layer
# of the neural network. Weights are THE MOST IMPORTANT PART of
# a neural network, as we tune these such that the network produces
# an accurate output. Biases are added to the output of each layer
# to shift the activation function linearly, and also prevent the 
# nodes of a neural network from zeroing out.
def createWeightAndBias(prevLayerNodesNum, currLayerNodesNum):
    weight = np.random.randn(currLayerNodesNum, prevLayerNodesNum) # curr * prev so that matmul works out such that output has same number of rows as nodes in current hidden layer
    bias = np.random.randn(currLayerNodesNum)

    return weight, bias