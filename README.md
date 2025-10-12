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
from model_setup import shuffle_dataset, ActivationFunc, CostFunc


# Data
input_data = np.array([
    [100, 50, 30],
    [200, 20, 10],
    [150, 10, 70]
])
output_data = np.array([1, 0, 1])

input_data, output_data = shuffle_dataset(input_data, output_data)

# Setup
model = Model(
    numInputNodes = 3,
    numHiddenLayerNodes = [7, 7, 7],
    numOutputNodes = 1,
    activationFunc = [ActivationFunc.SIGMOID, ActivationFunc.SIGMOID, ActivationFunc.RELU],
    normalize = True
)


# Train
model.train(
    inputs = input_data,
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
- Adding a learning rate scheduler
- Residual network and ResNet
- Batch normalization
- Neural network regularization
- Handle multi dimensional inputs
- Split train and test dataset, then train and predict in one go
- Data, weight, and gradient clipping
- Implement model metrics
- Apply other normalization techniques
- Rename this to a Neural net repo

# Resources
- Basic neural network concepts and math
  - [Learn to Build a Neural Network from Scratch](https://medium.com/@waadlingaadil/learn-to-build-a-neural-network-from-scratch-yes-really-cac4ca457efc#:~:text=1) by _Aadil Mallick_
  - [The Importance and Reasoning behind Activation Functions](https://towardsdatascience.com/the-importance-and-reasoning-behind-activation-functions-4dc00e74db41/) by _Zach Brodtman_
  - [Undestanding Cost Functions in Machine Learning: Types and Applications](https://medium.com/@anishnama20/understanding-cost-functions-in-machine-learning-types-and-applications-cd7d8cc4b47d) by _Anishnama_
  - [Backpropagation, intuitively | Deep Learning Chapter 3](https://www.youtube.com/watch?v=Ilg3gGewQ5U) by _3Blue1Brown_
  - [Backpropagation calculus | Deep Learning Chapter 4](https://www.youtube.com/watch?v=Ilg3gGewQ5U) by _3Blue1Brown_
- Data handling
  - [What is Shuffling the Data? A Guide for Students](https://medium.com/@sanjay_dutta/what-is-shuffling-the-data-a-guide-for-students-0f874572baf6) by _Sanjay Dutta_
  - [Why should the data be shuffled for machine learning tasks](https://datascience.stackexchange.com/questions/24511/why-should-the-data-be-shuffled-for-machine-learning-tasks) on StackExchange
  - [Feature Engineering: Scaling, Normalization and Standardization](https://www.geeksforgeeks.org/machine-learning/Feature-Engineering-Scaling-Normalization-and-Standardization/) on GeeksforGeeks
- Weight initialization
  - [Weight Initialization Techniques for Deep Neural Networks](https://www.geeksforgeeks.org/machine-learning/weight-initialization-techniques-for-deep-neural-networks/) on GeeksforGeeks
  - [Xavier initialization](https://www.geeksforgeeks.org/deep-learning/xavier-initialization/) on GeeksforGeeks
  - [Kaiming Initialization in Deep Learning](https://www.geeksforgeeks.org/deep-learning/kaiming-initialization-in-deep-learning/) on GeeksforGeeks