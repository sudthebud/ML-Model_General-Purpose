###########
# IMPORTS #
###########
import numpy as np

import model_setup



#############
# FUNCTIONS #
#############
class Model:

    def __init__(self, numInputNodes: int, numHiddenLayerNodes: list[int], numOutputNodes: int, activationFunc: int | list[int], costFunc: int):
        self.__numInputNodes = numInputNodes
        self.__numHiddenLayerNodes = numHiddenLayerNodes
        self.__numOutputNodes = numOutputNodes
        if isinstance(activationFunc, list[int]) and len(activationFunc) != len(numHiddenLayerNodes) + 1:
            raise ValueError("activationFunc list length must be same as the number of layers (except for the input layer)")
        else: self.__activationFunc = activationFunc
        self.__costFunc = costFunc

        self.__setup()

    # Create the list of weights and biases
    def __setup(self):
        self.weights = []
        self.biases = []
        
        for i in range(len(self.__numHiddenLayerNodes) + 1):
            weight, bias = model_setup.create_weight_and_bias(self.__numHiddenLayerNodes[i-1] if i > 0 else self.__numInputNodes,
                                                           self.__numHiddenLayerNodes[i] if i < len(self.__numHiddenLayerNodes) else self.__numOutputNodes)
            
            self.weights.append(weight)
            self.biases.append(bias)


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
            currLayer = self.weights[i] @ layers[-1]

            # Add the biases of each node in the layer. Numpy will
            # automatically broadcast the bias matrix if the number
            # of columns in the current layer is greater than 1.
            currLayer += self.biases[i]

            # Apply the activation function to all nodes in the 
            # current layer.
            currLayer = model_setup.activation(currLayer, self.__activationFunc if isinstance(self.__activationFunc, list[int]) else self.__activationFunc[i])

            layers.append(currLayer)
            if i == len(self.__numHiddenLayerNodes) + 1: return layers

    def __back_propagation(self, layers, expected_out):

        weightGradients, biasGradients = [], []
        
        # Derivative of cost function with respect to final
        # layer output result (which is the result of the
        # activation function).
        dC_dA = model_setup.cost_derivative(layers[-1], expected_out, self.__costFunc)
        chainedDerivatives = dC_dA
        
        for i in range(len(layers), 0): # For each layer in the neural network besides the input layer (going backwards)...

            dA_dZ = model_setup.activation_derivative(layers[i], self.__activationFunc)
            chainedDerivatives = chainedDerivatives @ dA_dZ.T

            dZ_dW = layers[i-1]
            weightGradient = chainedDerivatives @ dZ_dW.T
            weightGradients.insert(0, weightGradient)
            
            
