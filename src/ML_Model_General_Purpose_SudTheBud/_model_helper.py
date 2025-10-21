###########
# IMPORTS #
###########
from os import path
from zipfile import is_zipfile, ZipFile
from json import load
from shutil import rmtree
from sys import float_info
import math
from enum import IntEnum

import numpy as np


####################
# CONSTS AND ENUMS #
####################
_ATTRPATH = "attr.json"
_WEIGHTSPATH = "weights.npz"
_BIASESPATH = "biases.npz"
_NORMALIZATIONCACHEPATH = "normalizationCache.npz"
_STANDARDIZATIONCACHEPATH = "standardizationCache.npz"

_WEIGHT_AND_BIAS_RNG = np.random.default_rng()
class WeightInitFunc(IntEnum):
    RANDOM_UNIFORM = 0
    RANDOM_NORMAL = 1
    XAVIER_UNIFORM = 2
    XAVIER_NORMAL = 3
    HE_UNIFORM = 4
    HE_NORMAL = 5
_BIAS_SMALL_ALPHA_INIT = 0.01
class BiasInitFunc(IntEnum):
    ZERO = 0
    SMALL_ALPHA = 1
    RANDOM_NORMAL= 2

_LEAKYRELU_ALPHA = 0.01
class ActivationFunc(IntEnum):
    SIGMOID = 0
    TANH = 1
    RELU = 2
    LEAKY_RELU = 3
    SOFTMAX = 4

class CostFunc(IntEnum):
    MEAN_SQ_ERROR = 0
    # RT_MEAN_SQ_ERROR = 1
    MEAN_ABS_ERROR = 2
    BINARY_CROSS_ENTROPY = 3
    CATEGORICAL_CROSS_ENTROPY = 4
    # HINGE_LOSS = 5
    # KL_DIVERGENCE = 6

class LearningRateSchedulerFunc(IntEnum):
    CONSTANT = 0,
    STEP_DECAY = 1,
    EXPONENTIAL_DECAY = 2,
    COSINE_ANNEALING = 3



#############
# FUNCTIONS #
#############

# Load model from a ZIP file.
def load_model(filePath: str) -> 'Model':
    if not path.isfile(filePath) or not is_zipfile(filePath):
        raise ValueError("Invalid file path")

    
    # Load attributes and arrays
    with ZipFile(filePath) as zipFile:
        with zipFile.open(_ATTRPATH, 'r') as file: attr = load(file)
        weights = list(np.load(zipFile.open(_WEIGHTSPATH, 'r')).values())
        biases = list(np.load(zipFile.open(_BIASESPATH, 'r')).values())

        if _NORMALIZATIONCACHEPATH in zipFile.namelist():
            normalizationCache = np.load(zipFile.open(_NORMALIZATIONCACHEPATH, 'r'))
        else: normalizationCache = {'min': None, 'max': None}

        if _STANDARDIZATIONCACHEPATH in zipFile.namelist():
            standardizationCache = np.load(zipFile.open(_STANDARDIZATIONCACHEPATH, 'r'))
        else: standardizationCache = {'mean': None, 'stDev': None}


    # Create model and set attributes and arrays
    from ._model import Model
    model = Model(numInputNodes = attr['numInputNodes'],
                  numHiddenLayerNodes = attr['numHiddenLayerNodes'],
                  numOutputNodes = attr['numOutputNodes'],
                  activationFunc = attr['activationFunc'],
                  costFunc = attr['costFunc'],
                  weightInitFunc = attr['weightInitFunc'],
                  biasInitFunc = attr['biasInitFunc'],
                  normalize = attr['normalize'],
                  standardize = attr['standardize'])
    model._load_model_arrays(weights, biases, normalizationCache, standardizationCache)


    return model


# Function to shuffle training and output data together. Shuffling
# data is useful so that the model does not learn to recognize patterns
# in the data order or the model can "bounce out" of a local minimum
# of the cost function during training.
def shuffle_dataset(inputs: np.array, outputs: np.array) -> tuple[np.array, np.array, np.array]:
    if len(inputs.shape) != 2:
        if len(inputs.shape) == 1: inputs = inputs[np.newaxis, :]
        else: raise ValueError("Dimensions of predicted values must be 2D")
    if len(outputs.shape) != 2:
        if len(outputs.shape) == 1: outputs = outputs[np.newaxis, :]
        else: raise ValueError("Dimensions of output must be 2D")
    if inputs.shape[0] != outputs.shape[0]:
        raise ValueError("Number of training samples does not equal number of outputs")


    rng = np.random.default_rng()
    permutationIndices = rng.permutation(inputs.shape[0])

    inputs = inputs[permutationIndices, :]
    outputs = outputs[permutationIndices, :]

    # Thanks to @Ali on StackOverflow for this
    reversePermutationIndices = np.empty_like(permutationIndices)
    reversePermutationIndices[permutationIndices] = np.arange(permutationIndices.size)

    return inputs, outputs, reversePermutationIndices

# Normalize training data, and save the input metrics for normalization
# (e.g. min, max) to be cached for when we have to normalize prediction
# inputs by the same metrics. Normalization is useful to prevent features
# that are inherently going to be larger from skewing the output values
# of nodes in the machine learning model.
# 
# This method will only set the normalization metrics when the first set
# of training data is given to the model (a large batch of >1 samples of 
# training data). Currently, the method implemented is min-max normalization.
def normalizate_dataset(inputs, normalizationMin_CACHE, normalizationMax_CACHE):
    if len(inputs.shape) != 2:
        if len(inputs.shape) == 1: inputs = inputs[np.newaxis, :]
        else: raise ValueError("Dimensions of predicted values must be 2D")


    if normalizationMin_CACHE is None and normalizationMax_CACHE is None and inputs.shape[0] > 1:
        normalizationMin_CACHE = np.min(inputs, axis=0)[np.newaxis, :]
        normalizationMax_CACHE = np.max(inputs, axis=0)[np.newaxis, :]

    if normalizationMin_CACHE is not None and normalizationMin_CACHE is not None:
        with np.errstate(divide='ignore', invalid='ignore'): 
            normalizedInputs = np.where(normalizationMax_CACHE - normalizationMin_CACHE != 0, (inputs - normalizationMin_CACHE) / (normalizationMax_CACHE - normalizationMin_CACHE), 0)

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
def standardize_dataset(inputs, standardizationMean_CACHE, standardizationStDev_CACHE):
    if len(inputs.shape) != 2:
        if len(inputs.shape) == 1: inputs = inputs[np.newaxis, :]
        else: raise ValueError("Dimensions of predicted values must be 2D")


    if standardizationMean_CACHE is None and standardizationStDev_CACHE is None and inputs.shape[0] > 1:
        standardizationMean_CACHE = np.mean(inputs, axis=0)[np.newaxis, :]
        standardizationStDev_CACHE = np.std(inputs, axis=0)[np.newaxis, :]

    if standardizationMean_CACHE is not None and standardizationStDev_CACHE is not None:
        with np.errstate(divide='ignore', invalid='ignore'): 
            standardizedInputs = np.where(standardizationStDev_CACHE != 0, (inputs - standardizationMean_CACHE) / standardizationStDev_CACHE, 0)

        return standardizedInputs, standardizationMean_CACHE, standardizationStDev_CACHE
    else:
        return inputs, standardizationMean_CACHE, standardizationStDev_CACHE



# Initialize weights based on selected weight function. Weight
# initialization is important, since a poor set of initialized
# weights can lead to problems such as vanishing gradients
# (gradients get smaller) or exploding gradients (gradients get
# huge). There are several methods for weight initialization,
# the most simple being randomly sampling from a uniform or normal
# distribution, but more advanced methods such as Xavier or He
# initialization exist, which sample from a distribution based on
# the number of input and output nodes, which stabilizes the variance
# of the distribution no matter how many inputs and outputs there are.
def _weight_initialization(currLayerNodesNum, prevLayerNodesNum, inputNodes, outputNodes, weightInitFunc):
    weight = np.empty((currLayerNodesNum, prevLayerNodesNum)) # curr * prev so that matmul works out such that output has same number of rows as nodes in current hidden layer

    match weightInitFunc:
        case WeightInitFunc.RANDOM_UNIFORM:
            weight = _WEIGHT_AND_BIAS_RNG.uniform(-1, 1, weight.shape)

        case WeightInitFunc.RANDOM_NORMAL:
            weight = _WEIGHT_AND_BIAS_RNG.standard_normal(weight.shape)

        case WeightInitFunc.XAVIER_UNIFORM:
            distributionBound = (6 / (inputNodes + outputNodes)) ** 0.5
            weight = _WEIGHT_AND_BIAS_RNG.uniform(-distributionBound, distributionBound, weight.shape)

        case WeightInitFunc.XAVIER_NORMAL:
            stDev = (2 / (inputNodes + outputNodes)) ** 0.5
            weight = _WEIGHT_AND_BIAS_RNG.normal(0, stDev, weight.shape)

        case WeightInitFunc.HE_UNIFORM:
            distributionLowerBound = -(6 / inputNodes) ** 0.5
            distributionUpperBound = (6 / outputNodes) ** 0.5
            weight = _WEIGHT_AND_BIAS_RNG.uniform(distributionLowerBound, distributionUpperBound, weight.shape)

        case WeightInitFunc.HE_NORMAL:
            stDev = (2 / inputNodes) ** 0.5
            weight = _WEIGHT_AND_BIAS_RNG.normal(0, stDev, weight.shape)

        case _: raise ValueError("Invalid weight initialization function")

    return weight

# Initialize biases based on the selected bias function. Most
# times, the bias is initialized at zero since it matters less
# to the matrix calculations than weights, which can zero out
# the calculations if they are set to zero, but sometimes the
# bias can be set to a small value for instances like ReLU
# activation functions.
def _bias_initialization(currLayerNodesNum, biasInitFunc):
    bias = np.empty((currLayerNodesNum, 1))

    match biasInitFunc:
        case BiasInitFunc.ZERO: bias.fill(0)
        case BiasInitFunc.SMALL_ALPHA: bias.fill(_BIAS_SMALL_ALPHA_INIT)
        case BiasInitFunc.RANDOM_NORMAL: bias = _WEIGHT_AND_BIAS_RNG.standard_normal(bias.shape)

        case _: raise ValueError("Invalid bias initialization function")

    return bias



# Nonlinear activation function that converts neuron outputs.
# Necessary to introduce nonlinearity to neural network
# (otherwise the result of the network is basically the output
# of a giant linear function which is pretty useless for
# nonlinear problems).
def _activation(matrix, activationFunc):
    match activationFunc:
        case ActivationFunc.SIGMOID: return np.where(matrix < 0, np.exp(matrix) / (1 + np.exp(matrix)), 1 / (1 + np.exp(-matrix)))
        case ActivationFunc.TANH: return np.where(matrix < 0, (np.exp(2 * matrix) - 1) / (np.exp(2 * matrix) + 1), (1 - np.exp(-2 * matrix)) / (1 + np.exp(-2 * matrix)))
        case ActivationFunc.RELU: return np.where(matrix >= 0, matrix, 0)
        case ActivationFunc.LEAKY_RELU: return np.where(matrix >= 0, matrix, -_LEAKYRELU_ALPHA * matrix)
        case ActivationFunc.SOFTMAX: 
            with np.errstate(over='ignore'):
                matrixExpCatchZero = np.where(matrix >= 0, np.exp(-matrix), -1)
                matrixExpCatchZero = np.where(matrixExpCatchZero == 0, float_info.min, matrixExpCatchZero)
                matrixExp = np.where(matrix < 0, np.exp(matrix), 1/matrixExpCatchZero)
                return matrixExp / np.sum(matrixExp, axis = 0)

        case _: raise ValueError("Invalid activation function")

# Derivative of the activation functions with respect to
# neuron outputs. Used as part of chain rule in backpropagation.
def _activation_derivative(matrix, activationFunc):
    match activationFunc:
        case ActivationFunc.SIGMOID: return matrix * (1 - matrix)
        case ActivationFunc.TANH: return 1 - matrix ** 2
        case ActivationFunc.RELU: return np.where(matrix >= 0, 1, 0)
        case ActivationFunc.LEAKY_RELU: return np.where(matrix >= 0, 1, -_LEAKYRELU_ALPHA)

        case ActivationFunc.SOFTMAX:
            # Softmax is a little more complex, since we have to get the derivative
            # of every activated node with respect to the unactivated node output
            # (which is a_j(1-a_j) if a_i = a_j and a_i is the activated node in question,
            # and a_i * a_j if a_i is any other activated node in the layer). Then, we
            # need to multiply those values by the derivative of the cost with respect 
            # to every activated node in the layer, respectively (done in the backprop 
            # function in the Model class). Thus, we are required to make a n x m x m array,
            # then sum those products for each node later.
            matrixTiled = matrix.T[:, :, np.newaxis]
            matrixTiled = np.tile(matrixTiled, [1, 1, matrixTiled.shape[1]])

            matrixDot = -matrixTiled * matrix.T[:, np.newaxis, :]
            matrixDot[:, np.diag_indices(matrixDot.shape[1])[0], np.diag_indices(matrixDot.shape[2])[1]] += matrix.T
            matrixDot = matrixDot.T

            return matrixDot

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

        case CostFunc.CATEGORICAL_CROSS_ENTROPY:
            # 1 / n * sum(sum(y_a * log(y_p)))
            result = -np.sum((actual * np.log(predicted)), axis = 0)
            allResults = (1 / numTests) * np.sum(result)

        # case CostFunc.HINGE_LOSS:
        #     result = np.maximum(0, 1 - actual * predicted)
        #     allResults = np.mean(result, axis = 1)
        # case CostFunc.KL_DIVERGENCE:
        #     result = np.sum((actual * np.log(actual / predicted)), axis = 0)
        #     allResults = np.mean(result)


        case _: raise ValueError("Invalid cost function")

    cost = allResults
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
            result = (predicted - actual) / np.abs(predicted - actual)
            allResults = 1 / numTests * result
        
        case CostFunc.BINARY_CROSS_ENTROPY: 
            # 1 / n * sum((1 - y_a) / (1 - y_p) - y_a / y_p)
            result = (1 - actual) / (1 - predicted) - actual / predicted
            allResults = 1 / numTests * result

        case CostFunc.CATEGORICAL_CROSS_ENTROPY:
            # y_a / y_p
            result = -(actual / predicted)
            allResults = result


        case _: raise ValueError("Invalid cost function")

    costDerived = allResults
    return costDerived


# Update the learning rate according to a "scheduling" function,
# meaning we update the learning rate to a lower value as we move
# through a number of epochs in training. In the beginning of
# our training process, we want a high learning rate so that our
# parameters make large movements toward being tuned to provide
# the minimum of the cost function (and thus being as accurate
# as possible). However, as we keep running the training process,
# we will inevitably get our model closer to the minimum of the
# cost function, and that's when we want the learning rate to be
# small so we don't make big changes to our parameters and overshoot
# the minimum. A learning rate scheduler helps reduce the learning
# rate over time so that we can make big changes at the beginning
# of training and small changes at the end.
def _learning_rate_scheduler(epoch, 
                             maxEpochs, 
                             learningRateMax, 
                             learningRateFunc, 
                             learningRateMin = 0.01, 
                             learningRateStepSize = 10, 
                             learningRateDecayFactor = 0.05):
    
    match learningRateFunc:
        case LearningRateSchedulerFunc.CONSTANT: learningRate = learningRateMax
        case LearningRateSchedulerFunc.STEP_DECAY: learningRate = learningRateMax * (learningRateDecayFactor ** math.floor((1 + epoch) / learningRateStepSize))
        case LearningRateSchedulerFunc.EXPONENTIAL_DECAY: learningRate = learningRateMax * (math.e ** (-learningRateDecayFactor * epoch))
        case LearningRateSchedulerFunc.COSINE_ANNEALING: learningRate = learningRateMin + 0.5 * (learningRateMax - learningRateMin) * (1 + math.cos(epoch / maxEpochs * math.pi))

        case _: raise ValueError("Invalid learning rate scheduler function")

    return learningRate


# Calculate metrics for the output of a linear regression model.
# Linear regression models are models that predict a numerical value
# that can be among any range (such as predicting what someone's
# height is). Metrics, therefore, measure by how much the predicted
# values are off from their true values.
def regression_metrics(predicted: np.array, actual: np.array) -> tuple[np.array, np.array, np.array, np.array, np.array]:
    if not (actual == 0 or actual == 1).all(): raise ValueError("Values of actual (expected outputs) must be either 0 or 1")

    if len(predicted.shape) != 2:
        if len(predicted.shape) == 1: predicted = predicted[np.newaxis, :]
        else: raise ValueError("Dimensions of predicted values must be 2D")
    if len(actual.shape) != 2:
        if len(actual.shape) == 1: actual = actual[np.newaxis, :]
        else: raise ValueError("Dimensions of output must be 2D")
    if predicted.shape != actual.shape:
        raise ValueError("Shape of predicted values array does not equal shape of actual values array")
    
    numTests = actual.shape[0]


    # Residuals for all data points (actual - predicted)
    residual = actual - predicted


    # R^2 = 1 - sum(residual^2) / sum((actual - mean(actual))^2)
    r2 = 1 - np.sum(residual ** 2, axis = 0) / np.sum((actual - np.mean(actual, axis = 0)) ** 2, axis = 0)

    # Mean squared error = 1 / n * sum(residual^2)
    mse = 1 / numTests * np.sum(residual ** 2, axis = 0)

    # Root mean squared error = sqrt(1 / n * sum(residual ** 2))
    rmse = mse ** 0.5

    # Mean absolute error = 1 / n * sum(|residual|)
    mae = 1 / numTests * np.sum(np.abs(residual), axis = 0)

    # Mean absolute percentage error = 1 / n * sum(|residual / actual|)
    mape = 1 / numTests * np.sum(np.abs(residual / actual), axis = 0)


    # Remove axes if it is convenient
    for metric in [r2, mse, rmse, mae, mape]:
        if metric.shape == (1, 1): metric = metric[0, 0]
        elif metric.shape[0] == 1: metric = np.reshape(metric, metric.shape[1])
        elif metric.shape[1] == 1: metric = np.reshape(metric, metric.shape[0])
    
    return r2, mse, rmse, mae, mape

# Calculate metrics for the output of a classification model. 
# Classification models are models that predict whether the input 
# falls into a certain class (such as whether the object in an
# image is an apple), so for each class, their can only be a 
# positive "yes" or negative "no". Metrics, therefore, measure
# how many predictions are classified correctly.
def classification_metrics(predicted: np.array, actual: np.array, threshold: float = 0.5, multilabel: bool = False):
    if not (actual == 0 or actual == 1).all(): raise ValueError("Values of actual (expected outputs) must be either 0 or 1")

    if len(predicted.shape) != 2:
        if len(predicted.shape) == 1: predicted = predicted[np.newaxis, :]
        else: raise ValueError("Dimensions of predicted values must be 2D")
    if len(actual.shape) != 2:
        if len(actual.shape) == 1: actual = actual[np.newaxis, :]
        else: raise ValueError("Dimensions of output must be 2D")
    if predicted.shape != actual.shape:
        raise ValueError("Shape of predicted values array does not equal shape of actual values array")
    

    if multilabel or actual.shape[1] == 1:
        predicted = np.where(predicted > threshold, 1, 0) # Positive for class if probability is over threshold
    else:
        predictedTEMP = np.zeros_like(predicted)
        predictedTEMP[np.arange(predicted.shape[0]), np.argmax(predicted, axis = 1)] = 1

        predicted = predictedTEMP    


    # Accuracy = correct classifications / total
    accuracy = np.where(predicted == actual, 1, 0)
    accuracy = np.sum(accuracy, axis = 0) / actual.shape[0]

    # Recall = correct classified positives / total true positives
    recall = np.where(predicted == actual and actual == 1, 1, 0)
    recall = np.sum(recall, axis = 0) / np.sum(actual, axis = 0)

    # False positive = incorrect classified positives / total true negatives
    fpr = np.where(predicted != actual and actual == 0, 1, 0)
    fpr = np.sum(fpr, axis = 0) / (actual.shape[0] - np.sum(actual, axis = 0))

    # Precision = correct classified positives / all classified positives
    precision = np.where(predicted == actual and actual == 1, 1, 0)
    precision = np.sum(precision, axis = 0) / np.sum(predicted, axis = 0)


    # Remove axes if it is convenient
    for metric in [accuracy, recall, fpr, precision]:
        if metric.shape == (1, 1): metric = metric[0, 0]
        elif metric.shape[0] == 1: metric = np.reshape(metric, metric.shape[1])
        elif metric.shape[1] == 1: metric = np.reshape(metric, metric.shape[0])

    return accuracy, recall, fpr, precision