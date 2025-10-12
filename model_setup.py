###########
# IMPORTS #
###########
from enum import Enum

import numpy as np


####################
# CONSTS AND ENUMS #
####################
_leakyReLU_Alpha = 0.01
class ActivationFunc(Enum):
    SIGMOID = 0
    TANH = 1
    RELU = 2
    LEAKY_RELU = 3

class CostFunc(Enum):
    MEAN_SQ_ERROR = 0
    # RT_MEAN_SQ_ERROR = 1
    MEAN_ABS_ERROR = 2
    BINARY_CROSS_ENTROPY = 3
    # CATEGORICAL_CROSS_ENTROPY = 4
    # HINGE_LOSS = 5
    # KL_DIVERGENCE = 6



#############
# FUNCTIONS #
#############

# Function to shuffle training and output data together. Shuffling
# data is useful so that the model does not learn to recognize patterns
# in the data order or the model can "bounce out" of a local minimum
# of the cost function during training.
def shuffle_dataset(inputs, outputs):
    rng = np.random.default_rng()
    permutationIndices = rng.permutation(inputs.shape[1])

    inputs = inputs[:, permutationIndices]
    outputs = outputs[:, permutationIndices]

    return inputs, outputs

# Normalize training data, and save the input metrics for normalization
# (e.g. min, max) to be cached for when we have to normalize prediction
# inputs by the same metrics. Normalization is useful to prevent features
# that are inherently going to be larger from skewing the output values
# of nodes in the machine learning model.
# 
# This method will only set the normalization metrics when the first set
# of training data is given to the model (a large batch of >1 samples of 
# training data). Currently, the method implemented is min-max normalization.
def _normalization(inputs, normalizationMin_CACHE, normalizationMax_CACHE):
    if normalizationMin_CACHE is None and normalizationMax_CACHE is None and inputs.shape[1] > 1:
        normalizationMin_CACHE = np.min(inputs, axis=1)[:, np.newaxis]
        normalizationMax_CACHE = np.max(inputs, axis=1)[:, np.newaxis]

    if normalizationMin_CACHE is not None and normalizationMin_CACHE is not None:
        normalizedInputs = (inputs - normalizationMin_CACHE) / (normalizationMax_CACHE - normalizationMin_CACHE)

        return normalizedInputs, normalizationMin_CACHE, normalizationMax_CACHE
    else:
        return inputs, normalizationMin_CACHE, normalizationMax_CACHE

# Standardize / standard scale training data, and save the input metrics
# for standardization (mean and standard deviation) to be cached for when
# we have to standardize prediction inputs by the same metrics. Standardization
# is useful for the same reasons as normalization, except here, we scale
# feature values by the their variance from the mean. This method is less
# susceptible to outliers than min-max normalization and is useful
# for normally distributed data.
# 
# This method will only set the standardization metrics when the first set
# of training data is given to the model (a large batch of >1 samples of 
# training data).
def _standardization(inputs, standardizationMean_CACHE, standardizationStDev_CACHE):
    if standardizationMean_CACHE is None and standardizationStDev_CACHE is None and inputs.shape[1] > 1:
        standardizationMean_CACHE = np.mean(inputs, axis=1)[:, np.newaxis]
        standardizationStDev_CACHE = np.std(inputs, axis=1)[:, np.newaxis]

    if standardizationMean_CACHE is not None and standardizationStDev_CACHE is not None:
        standardizedInputs = (inputs - standardizationMean_CACHE) / standardizationStDev_CACHE

        return standardizedInputs, standardizationMean_CACHE, standardizationStDev_CACHE
    else:
        return inputs, standardizationMean_CACHE, standardizationStDev_CACHE



# Nonlinear activation function that converts neuron outputs.
# Necessary to introduce nonlinearity to neural network
# (otherwise the result of the network is basically the output
# of a giant linear function which is pretty useless for
# nonlinear problems).
def _activation(matrix, activationFunc):
    match activationFunc:
        case ActivationFunc.SIGMOID: return 1 / (1 + np.exp(-matrix))
        case ActivationFunc.TANH: return (np.exp(matrix) - np.exp(-matrix)) / (np.exp(matrix) + np.exp(-matrix))
        case ActivationFunc.RELU: return np.where(matrix >= 0, matrix, 0)
        case ActivationFunc.LEAKY_RELU: return np.where(matrix >= 0, matrix, -_leakyReLU_Alpha * matrix)

        case _: raise ValueError("Invalid activation function")

# Derivative of the activation functions with respect to
# neuron outputs. Used as part of chain rule in backpropagation.
def _activation_derivative(matrix, activationFunc):
    match activationFunc:
        case ActivationFunc.SIGMOID: return matrix * (1 - matrix)
        case ActivationFunc.TANH: return 1 - matrix ** 2
        case ActivationFunc.RELU: return np.where(matrix >= 0, 1, 0)
        case ActivationFunc.LEAKY_RELU: return np.where(matrix >= 0, 1, -_leakyReLU_Alpha) 

        case _: raise ValueError("Invalid activation function")



# Different cost functions applicable in different modeling
# situations. Even if multiple vectorized training or test
# cases are run at once, this will compile the cost into a
# scalar result.
def _cost(predicted, actual, costFunc):
    numTests = predicted.shape[1]

    match costFunc:
        case CostFunc.MEAN_SQ_ERROR: 
            # 1 / n * sum((y_p - y_a)^2) 
            result = (predicted - actual) ** 2
            allResults = 1 / numTests * np.sum(result, axis = 1)

        # case CostFunc.RT_MEAN_SQ_ERROR: 
        #     # sqrt(1 / n * sum((y_p - y_a)^2))
        #     result = (predicted - actual) ** 2
        #     allResults = (1 / numTests * np.sum(result, axis = 1)) ** 0.5

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

    cost = allResults if len(allResults.shape) == 1 else np.sum(allResults) # Flatten the result to a scalar value to make it easier to work with when comparing epochs
    return cost

# Derivative of the cost functions with respect to predicted
# value. Used as part of chain rule in backpropagation. We are
# ignoring the summation here - using vectorization means we will
# automatically sum the necessary values when performing the back
# propagation matrix multiplications, as every value in a row/column
#  gets summed by nature of matrix multiplications.
def _cost_derivative(predicted, actual, costFunc):
    numTests = predicted.shape[1]

    match costFunc:
        case CostFunc.MEAN_SQ_ERROR: 
            # 1 / n * sum(2 * (y_a - y_p))
            result = 2 * (actual - predicted)
            allResults =  1 / numTests * result

        # case CostFunc.RT_MEAN_SQ_ERROR: 
        #     # 1 / 2 * 1 / sqrt(1 / n * sum((y_p - y_a)^2))
        #     resultNotDerived = (predicted - actual) ** 2
        #     result = 2 * (actual - predicted)
        #     allResults = 1 / (2 * (1 / numTests * resultNotDerived) ** 0.5) * 1 / numTests * result # This function doesn't work with vectorization, need to redo somehow
        
        case CostFunc.MEAN_ABS_ERROR: 
            # 1 / n * sum((y_p - y_a) / |y_p - y_a|)
            result = (predicted - actual) / abs(predicted - actual)
            allResults = 1 / numTests * result
        
        case CostFunc.BINARY_CROSS_ENTROPY: 
            # 1 / n * sum((1 - y_a) / (1 - y_p) - y_a / y_p)
            result = (1 - actual) / (1 - predicted) - actual / predicted
            allResults = 1 / numTests * result


        case _: raise ValueError("Invalid cost function")

    costDerived = allResults
    return costDerived