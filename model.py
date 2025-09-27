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
    # and return the output.
    def feed_forward(input):
        