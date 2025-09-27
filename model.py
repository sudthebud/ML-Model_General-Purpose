###########
# IMPORTS #
###########
import numpy as np

import model_setup



#############
# FUNCTIONS #
#############
class Model:

    def __init__(self, numInputNodes: int, numHiddenLayerNodes: list[int], numOutputNodes: int, activationFunc: int | list[int]):
        self.numInputNodes = numInputNodes
        self.numHiddenLayerNodes = numHiddenLayerNodes
        self.numOutputNodes = numOutputNodes
        if isinstance(activationFunc, list[int]) and len(activationFunc) != len(numHiddenLayerNodes) + 1:
            raise ValueError("activationFunc list length must be same as the number of layers (except for the input layer)")
        self.activationFunc = activationFunc

        self.setup()

    # Create the list of weights and biases
    def setup(self):
        self.weights = []
        self.biases = []
        
        for i in range(len(self.numHiddenLayerNodes) + 1):
            weight, bias = model_setup.create_weight_and_bias(self.numHiddenLayerNodes[i-1] if i > 0 else self.numInputNodes,
                                                           self.numHiddenLayerNodes[i] if i < len(self.numHiddenLayerNodes) else self.numOutputNodes)
            
            self.weights.append(weight)
            self.biases.append(bias)


    # Run the data through the layers of the neural network,
    # and return the output. Input can be more than 1 column,
    # which allows for vectorization of the input data (enabling
    # fast processing of the input and output using matrix math 
    # rather than looping.
    def feed_forward(self, input):
        for i in range(len(self.numHiddenLayerNodes) + 1): # For each layer in the neural network...
            
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
            currLayer = self.weights[i] @ input

            # Add the biases of each node in the layer. Numpy will
            # automatically broadcast the bias matrix if the number
            # of columns in the current layer is greater than 1.
            currLayer += self.biases[i]

            # Apply the activation function to all nodes in the 
            # current layer.
            currLayer = model_setup.activation(currLayer, self.activationFunc if isinstance(self.activationFunc, list[int]) else self.activationFunc[i])

            if i == len(self.numHiddenLayerNodes) + 1: return currLayer
            input = currLayer