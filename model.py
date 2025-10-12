###########
# IMPORTS #
###########
import numpy as np

import model_setup



#############
# FUNCTIONS #
#############
class Model:

    def __init__(self, 
                 numInputNodes: int, 
                 numHiddenLayerNodes: list[int], 
                 numOutputNodes: int, 
                 normalize: bool = False,
                 standardize: bool = False,
                 activationFunc: int | list[int] = model_setup.ActivationFunc.SIGMOID,
                 weightInitFunc: int | list[int] = model_setup.WeightInitFunc.RANDOM_UNIFORM,
                 biasInitFunc: int | list[int] = model_setup.BiasInitFunc.ZERO):
        
        self.__numInputNodes = numInputNodes
        self.__numHiddenLayerNodes = numHiddenLayerNodes
        self.__numOutputNodes = numOutputNodes

        if isinstance(weightInitFunc, list) and len(weightInitFunc) != len(numHiddenLayerNodes) + 1:
            raise ValueError("weightInitFunc list length must be the same as the number of layers (except for the input layer)")
        else: self.__weightInitFunc = weightInitFunc
        if isinstance(biasInitFunc, list) and len(biasInitFunc) != len(numHiddenLayerNodes) + 1:
            raise ValueError("biasInitFunc list length must be the same as the number of layers (except for the input layer)")
        else: self.__biasInitFunc = biasInitFunc

        if normalize and standardize: raise ValueError("Cannot normalize and standardize data at the same time")
        self.__normalize = normalize
        self.__standardize = standardize

        if isinstance(activationFunc, list) and len(activationFunc) != len(numHiddenLayerNodes) + 1:
            raise ValueError("activationFunc list length must be the same as the number of layers (except for the input layer)")
        else: self.__activationFunc = activationFunc


        self.__setup()

    # Create weights and biases for each hidden layer + output layer
    # of the neural network. Weights are THE MOST IMPORTANT PART of
    # a neural network, as we tune these such that the network produces
    # an accurate output. Biases are added to the output of each layer
    # to shift the activation function linearly, and also prevent the 
    # nodes of a neural network from zeroing out.
    # 
    # Additionally, set up other variables for the machine learning model.
    def __setup(self):
        # Set weights
        self.__weights = []
        self.__biases = []
        
        for i in range(len(self.__numHiddenLayerNodes) + 1): # For every hidden layer + the output layer...
            prevLayerNodesNum = self.__numHiddenLayerNodes[i-1] if i > 0 else self.__numInputNodes
            currLayerNodesNum = self.__numHiddenLayerNodes[i] if i < len(self.__numHiddenLayerNodes) else self.__numOutputNodes
            
            # Creates weight and bias matrices, with weight being curr * prev and bias being curr * 1
            weight = model_setup._weight_initialization(currLayerNodesNum, prevLayerNodesNum, self.__numInputNodes, self.__numOutputNodes, self.__weightInitFunc[i] if isinstance(self.__weightInitFunc, list) else self.__weightInitFunc)
            bias = model_setup._bias_initialization(currLayerNodesNum, self.__biasInitFunc[i] if isinstance(self.__biasInitFunc, list) else self.__biasInitFunc)
            
            self.__weights.append(weight)
            self.__biases.append(bias)


        # Set cached variables for normalization and standardization
        self.__normalizationMin_CACHE = None
        self.__normalizationMax_CACHE = None

        self.__standardizationMean_CACHE = None
        self.__standardizationStDev_CACHE = None


    # Run the data through the layers of the neural network,
    # and return the output. Input can be more than 1 column,
    # which allows for vectorization of the input data (enabling
    # fast processing of the input and output using matrix math 
    # rather than looping.
    def __feed_forward(self, input):
        layers = [input]

        for i in range(len(self.__numHiddenLayerNodes) + 1): # For each layer in the neural network...
            
            # Multiply the weights for that layer by the input
            # (which is either the input for the model or the)
            # previous layer. For each node in the layer, we 
            # multiply all the nodes in the previous layer
            # with the weight going into the node in the current
            # layer, then add them together. We do it in the order
            # of "weights * input" so that the result comes out
            # to the number of nodes in the current layer (so if
            # the input is 2 x 1 and the number of nodes in the
            # current layer is 3, then the number of weights is
            # 3 (for current layer nodes) x 2 (for previous layer
            # nodes), and the result is a 3 x 1 matrix.
            currLayer = self.__weights[i] @ layers[-1]

            # Add the biases of each node in the layer. Numpy will
            # automatically broadcast the bias matrix if the number
            # of columns in the current layer is greater than 1.
            currLayer += self.__biases[i]

            # Apply the activation function to all nodes in the 
            # current layer.
            currLayer = model_setup._activation(currLayer, self.__activationFunc[i] if isinstance(self.__activationFunc, list) else self.__activationFunc)

            layers.append(currLayer)

        return layers

    # Use the resulting predictions from the feed forward run
    # to update the weights and biases of the model. This is done
    # via gradient descent, where we use the derivative chain rule
    # to determine the effect that a single weight has on the cost
    # function, then change the weights and biases by a small amount
    # based on that effect. Running this in tandem with feed forward
    # many times results in a model whose weights and biases are
    # slowly tuned to output the desired result with a high level
    # of precision (if the model parameters are also tuned well).
    def __back_propagation(self, layers, expectedOut, costFunc, learningRate):

        weightGradients, biasGradients = [], []
        
        # Derivative of cost function with respect to final
        # layer output result (which is the result of the
        # activation function).
        dC_dA = model_setup._cost_derivative(layers[-1], expectedOut, costFunc)
        chainedDerivatives = dC_dA
        
        for i in range(len(layers)-1, 0, -1): # For each layer in the neural network besides the input layer (going backwards)...

            # Compute derivative of activation function with respect to neuron output
            dA_dZ = model_setup._activation_derivative(layers[i], self.__activationFunc)
            chainedDerivatives = chainedDerivatives * dA_dZ

            # Compute derivative of neuron output with respect to weights
            # and use chain derivatives to compute derivative of cost with
            # respect to weights for this layer
            dZ_dW = layers[i-1]
            weightGradient = chainedDerivatives @ dZ_dW.T
            weightGradients.insert(0, weightGradient)

            # Compute derivative of neuron output with respect to biases
            # and use chain derivatives to compute derivative of cost with
            # respect to biases for this layer (dZ_dB = 1, so no need to 
            # calculate the derivative itself)
            biasGradient = np.sum(chainedDerivatives, axis = 1, keepdims=True) # keepdims keeps the reduced dimension so it stays a 2D array
            biasGradients.insert(0, biasGradient)

            # Compute derivative of neuron output with respect to activation function
            # of LAST layer
            dZ_dA = self.__weights[i-1]
            chainedDerivatives = dZ_dA.T @ chainedDerivatives

        
        for i in range(len(layers) - 1): # For each layer in the neural network besides the output layer...

            # Subtract a small amount of the layer's weights' gradients from
            # the layers' weights. We subtract because dC_dW is a measure of 
            # how much one weight increases the result of the cost function - 
            # so, if we decrease the value of the weight, we decrease the result 
            # of the cost function, and vice versa if the gradient decreases
            # the result of the function. The gradient tells us the steepness
            # as to how much the weight increases or decreases the resulting cost
            # function, and allows us to update our weights by a factor of this 
            # slope accordingly, giving large decreases/increases for weights
            # that are a much bigger factor in the cost function's result.
            # We use a gradient factor (learning rate) so that the model
            # doesn't update too drastically at once, but it has to be fine
            # tuned so that it doesn't make gradient descent useless either.
            self.__weights[i] -= learningRate * weightGradients[i]

            # Do the same thing for biases
            self.__biases[i] -= learningRate * biasGradients[i]

    # Train the model by running feed forward and back propagation in
    # succession for several iterations (epochs). Feed forward will
    # calculate the value of each node in each layer; then, back propagation
    # will determine how much every weight and bias had an effect on making
    # the feed forward output layer result as incorrect as it is (using a
    # cost function that we define) and update the weights and biases
    # accordingly, by a factor of the learning rate.
    def train(self, inputs: np.array, expectedOut: np.array, epochs: int = 1000, costFunc: int = model_setup.CostFunc.BINARY_CROSS_ENTROPY, learningRate: float = 0.1):
        if len(inputs.shape) != 2:
            if len(inputs.shape) == 1: inputs = inputs[np.newaxis, :]
            else: raise ValueError("Dimensions of input must be 2D")
        if len(expectedOut.shape) != 2:
            if len(expectedOut.shape) == 1: 
                if inputs.shape[0] == 1 and expectedOut.shape[0] == self.__numOutputNodes: expectedOut = expectedOut[np.newaxis, :]
                else: expectedOut = expectedOut[:, np.newaxis]
            else: raise ValueError("Dimensions of output must be 2D")
        if inputs.shape[0] != expectedOut.shape[0]:
            raise ValueError("Number of training samples does not equal number of outputs")

        # Transpose into n features x m samples
        inputs = inputs.T
        expectedOut = expectedOut.T

        if inputs.shape[0] != self.__numInputNodes:
            raise ValueError("Number of input features must match value set for numInputNodes")
        if expectedOut.shape[0] != self.__numOutputNodes:
            raise ValueError("Number of output features must match value set for numOutputNodes")
        if epochs <= 0:
            raise ValueError("Number of epochs must be greater than 0")
        
        # Shuffle training data
        inputs, expectedOut = model_setup.shuffle_dataset(inputs, expectedOut)

        # Normalize training data
        if self.__normalize: inputs, self.__normalizationMin_CACHE, self.__normalizationMax_CACHE = model_setup._normalization(inputs, self.__normalizationMin_CACHE, self.__normalizationMax_CACHE)
        # Standardize training data
        elif self.__standardize: inputs, self.__standardizationMean_CACHE, self.__standardizationStDev_CACHE = model_setup._standardization(inputs, self.__standardizationMean_CACHE, self.__standardizationStDev_CACHE)

        
        costs = []

        for _ in range(epochs): # For every iteration (epoch)...

            # Run the feed forward process and return the resulting node values for every layer 
            layers = self.__feed_forward(inputs)

            # Determine the cost of this epoch (measure of how wrong the model was)
            # Not used in feed forward or back propagation, but to determine accuracy
            # of the model per epoch and adjust number of epochs accordingly (don't need
            # to run 1000 epochs if the model reaches a good accuracy at 50 epochs)
            cost = model_setup._cost(layers[-1], expectedOut, costFunc)
            costs.append(cost)

            # Run the back propagation process to update the weights and biases
            # based on how much they affect the result of the cost function (and
            # thus the output's accuracy)
            self.__back_propagation(layers, expectedOut, costFunc, learningRate)

        return costs 

    # Use the model on any number of data points. This method will run the 
    # feed forward algorithm on the input (only once since we have used
    # vectorization to calculate the result of all inputs at once), then
    # return the result of the feed forward process, with the outputs nodes
    # reflecting the predicted quantitative values (for linear regression)
    # or likelihood of the result falling into one or more classes (for 
    # classification).
    def predict(self, inputs: np.array):
        if len(inputs.shape) != 2:
            if len(inputs.shape) == 1: inputs = inputs[np.newaxis, :]
            else: raise ValueError("Dimensions of input must be 2D")

        # Transpose into n features x m samples
        inputs = inputs.T

        if inputs.shape[0] != self.__numInputNodes:
            raise ValueError("Number of input features must match value set for numInputNodes")
        
        # Normalize prediction input
        if self.__normalize: inputs, _, _ = model_setup._normalization(inputs, self.__normalizationMin_CACHE, self.__normalizationMax_CACHE)
        # Standardize prediction input
        elif self.__standardize: inputs, _, _ = model_setup._standardization(inputs, self.__standardizationMean_CACHE, self.__standardizationStDev_CACHE)

        
        # Run the feed forward algorithm and return the output layer
        layers = self.__feed_forward(inputs)
        predicted = layers[-1]

        # Transpose into m samples x n features, and remove axes if it is convenient
        predicted = predicted.T
        if predicted.shape == (1, 1): predicted = predicted[0, 0]
        elif predicted.shape[0] == 1: predicted = np.reshape(predicted, predicted.shape[1])
        elif predicted.shape[1] == 1: predicted = np.reshape(predicted, predicted.shape[0])

        return predicted