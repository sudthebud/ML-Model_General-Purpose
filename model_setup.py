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

class CostFunc(Enum):
    MEAN_SQ_ERROR = 0
    RT_MEAN_SQ_ERROR = 1
    MEAN_ABS_ERROR = 2
    BINARY_CROSS_ENTROPY = 3
    # CATEGORICAL_CROSS_ENTROPY = 4
    # HINGE_LOSS = 5
    # KL_DIVERGENCE = 6



#############
# FUNCTIONS #
#############

# Nonlinear activation function that converts neuron outputs.
# Necessary to introduce nonlinearity to neural network
# (otherwise the result of the network is basically the output
# of a giant linear function which is pretty useless for
# nonlinear problems).
def activation(matrix, activationFunc):
    match activationFunc:
        case ActivationFunc.SIGMOID: return 1 / (1 + np.exp(-matrix))
        case ActivationFunc.TANH: return (np.exp(matrix) - np.exp(-matrix)) / (np.exp(matrix) + np.exp(-matrix))
        case ActivationFunc.RELU: return np.where(matrix >= 0, matrix, 0)
        case ActivationFunc.LEAKY_RELU: return np.where(matrix >= 0, matrix, -leakyReLU_Alpha * matrix)

        case _: raise ValueError("Invalid activation function")

# Derivative of the activation functions with respect to
# neuron outputs. Used as part of chain rule in backpropagation.
def activation_derivative(matrix, activationFunc):
    match activationFunc:
        case ActivationFunc.SIGMOID: return matrix * (1 - matrix)
        case ActivationFunc.TANH: return 1 - matrix ** 2
        case ActivationFunc.RELU: return np.where(matrix >= 0, 1, 0)
        case ActivationFunc.LEAKY_RELU: return np.where(matrix >= 0, 1, -leakyReLU_Alpha) 

        case _: raise ValueError("Invalid activation function")



# Different cost functions applicable in different modeling
# situations. Even if multiple vectorized training or test
# cases are run at once, this will compile the cost into a
# scalar result.
def cost(predicted, actual, costFunc):
    numTests = predicted.shape[1]

    match costFunc:
        case CostFunc.MEAN_SQ_ERROR: 
            # 1 / n * sum((y_p - y_a)^2) 
            result = (predicted - actual) ** 2
            allResults = 1 / numTests * np.sum(result, axis = 1)

        case CostFunc.RT_MEAN_SQ_ERROR: 
            # sqrt(1 / n * sum((y_p - y_a)^2))
            result = (predicted - actual) ** 2
            allResults = (1 / numTests * np.sum(result, axis = 1)) ** 0.5

        case CostFunc.MEAN_ABS_ERROR: 
            # 1 / n * sum(|y_p - y_a|)
            result = abs(predicted - actual)
            allResults = 1 / numTests * np.sum(result, axis = 1)

        case CostFunc.BINARY_CROSS_ENTROPY: 
            # 1 / n * sum(-(log(y_p) * y_a + (1-y_a) * log(1-y_p)))
            result = -(actual * np.log(predicted) + (1-actual) * np.log(1-predicted))
            allResults = (1 / numTests) * np.sum(result, axis = 1)

        # case CostFunc.CATEGORICAL_CROSS_ENTROPY:
        #     result = -np.sum((actual * np.log(predicted)), axis = 0)
        #     allResults = (1 / numTests) * np.sum(result)
        # case CostFunc.HINGE_LOSS:
        #     result = np.maximum(0, 1 - actual * predicted)
        #     allResults = np.mean(result, axis = 1)
        # case CostFunc.KL_DIVERGENCE:
        #     result = np.sum((actual * np.log(actual / predicted)), axis = 0)
        #     allResults = np.mean(result)


        case _: raise ValueError("Invalid cost function")

    cost = allResults if len(allResults.shape) == 1 else np.sum(allResults) # Flatten the result to a scalar value to make it easier to work with
    return cost

# Derivative of the cost functions with respect to predicted
# value. Used as part of chain rule in backpropagation.
def cost_derivative(predicted, actual, costFunc):
    numTests = predicted.shape[1]

    match costFunc:
        case CostFunc.MEAN_SQ_ERROR: 
            # 1 / n * sum(2 * (y_a - y_p))
            result = 2 * (actual - predicted)
            allResults =  1 / numTests * np.sum(result, axis = 1)

        case CostFunc.RT_MEAN_SQ_ERROR: 
            # 1 / 2 * 1 / sqrt(1 / n * sum((y_p - y_a)^2))   *   1 / n * sum(2 * (y_a - y_p))
            resultNotDerived = (predicted - actual) ** 2
            result = 2 * (actual - predicted)
            allResults = 1 / (2 * (1 / numTests * np.sum(resultNotDerived, axis = 1)) ** 0.5) * 1 / numTests * np.sum(result, axis = 1)
        
        case CostFunc.MEAN_ABS_ERROR: 
            # 1 / n * sum((y_p - y_a) / |y_p - y_a|)
            result = (predicted - actual) / abs(predicted - actual)
            allResults = 1 / numTests * np.sum(result, axis = 1)
        
        case CostFunc.BINARY_CROSS_ENTROPY: 
            # 1 / n * sum(y_a / y_p + (1 - y_a) / (1 - y_p))
            result = actual / predicted + (1 - actual) / (1 - predicted)
            allResults = 1 / numTests * np.sum(result, axis = 1)


        case _: raise ValueError("Invalid cost function")

    cost = allResults if len(allResults.shape) == 1 else np.sum(allResults) # Flatten the result to a scalar value to make it easier to work with
    return cost

    

# Create weights and biases for each hidden layer + output layer
# of the neural network. Weights are THE MOST IMPORTANT PART of
# a neural network, as we tune these such that the network produces
# an accurate output. Biases are added to the output of each layer
# to shift the activation function linearly, and also prevent the 
# nodes of a neural network from zeroing out.
def create_weight_and_bias(prevLayerNodesNum, currLayerNodesNum):
    weight = np.random.randn(currLayerNodesNum, prevLayerNodesNum) # curr * prev so that matmul works out such that output has same number of rows as nodes in current hidden layer
    bias = np.random.randn(currLayerNodesNum)

    return weight, bias