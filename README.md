# General Purpose ML Model

A general purpose machine learning model written in Python, developed as a learning project to refresh my brain on machine learning and neural network principles. This project was aided by a learning resource by Aadil Mallick along with other resources (linked in credits), but the code was written by myself and with as much of an understanding as possible of the concepts behind what I was coding and how it played into the neural network process (as evidenced by the lengthy comments in the code). Basically, I tried to understand how neural networks worked and the math behind them, and wrote the code from there, rather than copying code or instructions blindly.

This model is configurable, allowing a user who imports these Python scripts to use any number of layers, as well as choosing the number of nodes for every single layer, the activation function for every layer, the cost function, and the learning rate.

## Installation
- Install the ```numpy``` Python package
- Clone this repository into your computer
- Move ```model.py``` and ```model_setup.py``` to your project folder
- Import ```model``` and ```model_setup``` in your Python project

## Example Usage

```python
# Imports
from model import Model
from model_setup import ActivationFunc, CostFunc


# Data
input_data = np.array([
    [100, 50, 30],
    [200, 20, 10],
    [150, 10, 70]
])
output_data = np.array([1, 0, 1])

# Setup
model = Model(
    numInputNodes = 3,
    numHiddenLayerNodes = [7, 7, 7],
    numOutputNodes = 1,
    activationFunc = [ActivationFunc.SIGMOID, ActivationFunc.SIGMOID, ActivationFunc.RELU]
)


# Train
model.train(
    inputs = input,
    expectedOut = output_data,
    epochs = 10,
    costFunc = CostFunc.BINARY_CROSS_ENTROPY,
    learningRate = 0.5
)

# Test/Predict
prediction = model.predict(
    inputs = np.array([[125, 40, 30]])
)
```

## Things to Add / Experiment With
- Allowing 1D single inputs for testing and prediction
- Change weight and bias initialization
- Adding a learning rate scheduler
- Data normalization
- Residual network and ResNet
- Batch normalization
- Training data shuffling
- Neural network regularization

# Credits
- [Learn to Build a Neural Network from Scratch](https://medium.com/@waadlingaadil/learn-to-build-a-neural-network-from-scratch-yes-really-cac4ca457efc#:~:text=1) by Aadil Mallick
- [The Importance and Reasoning behind Activation Functions](https://towardsdatascience.com/the-importance-and-reasoning-behind-activation-functions-4dc00e74db41/) by Zach Brodtman
- [Undestanding Cost Functions in Machine Learning: Types and Applications](https://medium.com/@anishnama20/understanding-cost-functions-in-machine-learning-types-and-applications-cd7d8cc4b47d) by Anishnama
- [Backpropagation, intuitively | Deep Learning Chapter 3](https://www.youtube.com/watch?v=Ilg3gGewQ5U) by 3Blue1Brown
- [Backpropagation calculus | Deep Learning Chapter 4](https://www.youtube.com/watch?v=Ilg3gGewQ5U) by 3Blue1Brown