# General Purpose ML Model Package

A **general purpose machine learning model (specifically neural network) package** written in Python, developed as a learning project for machine learning/neural network basics and Python package creation. Can be downloaded, installed, and imported by users in order to easily create custom neural network models with varying architectures for their own applications. Used a variety of resources (linked in the [resources](#resources) section) as a _guide_ while writing the neural network functionality from scratch.

**View the [implementations](#implementations) section** to see repositories that have implemented this package to create neural networks trained on certain datasets or for certain tasks.

## Functionality
- Perform **basic data preprocessing**, such as normalization and data shuffling
- Create models with **custom architectures** or **load pre-existing models**
  - Set your own number of layers and neurons for each layer
  - Set activation functions for each layer
  - Change weight and bias initialization functions
- Train models with **custom training parameters**
  - Set number of epochs, epoch logging, and model training outupt
  - Separate data into batches
  - Set cost function for final layer
  - Set learning rate, as well as learning rate scheduler function
  - Clip gradients to mitigate exploding/vanishing gradients problem
- Test model on **test data or your own inputs** to predict
- Perform **classification or regression metrics** on model outputs
- Able to handle **any combination** of activation functions, cost functions, and other model parameters
- **Vectorized data** substantially improves model performance
- **Save model** to be used again

## Installation
1. Clone this repository into your computer
2. In your terminal, change the working directory to your clone of this repository
3. Run ```py -m build```
4. Activate the virtual environment of the project that will use this package
5. Run ```pip install [path\to\whl\file\in\dist]``` with the **.whl** file that gets created in the ```dist``` folder of this repository
6. Import the package

## Example Usage

```python
# Imports
import numpy as np

from ML_Model_General_Purpose_SudTheBud import (
  # Dataset processing
  shuffle_dataset, normalizate_dataset, standardize_dataset, 
  # Model creation
  Model, load_model, 
  # Model enums
  WeightInitFunc, BiasInitFunc, ActivationFunc, CostFunc, LearningRateSchedulerFunc, 
  # Prediction metrics
  regression_metrics, classification_metrics
)


# Data
input_data = np.array([
    [100, 50, 30],
    [200, 20, 10],
    [150, 10, 70]
])
output_data = np.array([1, 0, 1])

input_data, output_data, _ = shuffle_dataset(input_data, output_data)

# Setup
model = Model(
    numInputNodes = 3,
    numHiddenLayerNodes = [7, 7, 7],
    numOutputNodes = 1,
    activationFunc = [ActivationFunc.SIGMOID, ActivationFunc.SIGMOID, ActivationFunc.RELU],
    costFunc = CostFunc.BINARY_CROSS_ENTROPY,
    normalize = True
)


# Train
model.train(
    inputs = input_data,
    expectedOut = output_data,
    epochs = 10,
    learningRate = 0.5
)

# Test/Predict
prediction = model.predict(
    inputs = np.array([[125, 40, 30]])
)

# Metrics
accuracy, recall, fpr, precision, f1 = classification_metrics(prediction, actual)
```

## Things to Add / Experiment With
- Batch normalization
- Regularization
- Handle multi dimensional inputs
- Method to split train and test dataset, then train and test in one go
- Data and weight clipping
- Apply other normalization techniques
- Fix overflow and invalid value errors
- Rename this to a Neural net repo

## Resources
- Basic neural network concepts and math
  - [Learn to Build a Neural Network from Scratch](https://medium.com/@waadlingaadil/learn-to-build-a-neural-network-from-scratch-yes-really-cac4ca457efc#:~:text=1) by _Aadil Mallick_
  - [Backpropagation, intuitively | Deep Learning Chapter 3](https://www.youtube.com/watch?v=Ilg3gGewQ5U) by _3Blue1Brown_
  - [Backpropagation calculus | Deep Learning Chapter 4](https://www.youtube.com/watch?v=Ilg3gGewQ5U) by _3Blue1Brown_
  - [CS231n Deep Learning for Computer Vision](https://cs231n.github.io/neural-networks-2/) from Stanford University
- Data handling
  - [What is Shuffling the Data? A Guide for Students](https://medium.com/@sanjay_dutta/what-is-shuffling-the-data-a-guide-for-students-0f874572baf6) by _Sanjay Dutta_
  - [Why should the data be shuffled for machine learning tasks](https://datascience.stackexchange.com/questions/24511/why-should-the-data-be-shuffled-for-machine-learning-tasks) on StackExchange
  - [Feature Engineering: Scaling, Normalization and Standardization](https://www.geeksforgeeks.org/machine-learning/Feature-Engineering-Scaling-Normalization-and-Standardization/) on GeeksforGeeks
  - [Batching and Mini-Batch: Making Your Deep Learning Model Work Efficiently](https://medium.com/@nasuhcanturker/batching-and-mini-batch-making-your-deep-learning-model-work-efficiently-1bb5d3481eda) by _NasuhcaN_
  - [Why do large mini-batch sizes adversely affect validation accuracy?](https://www.quora.com/Why-do-large-mini-batch-sizes-adversely-affect-validation-accuracy) on Quora
- Initialization
  - [Weight Initialization Techniques for Deep Neural Networks](https://www.geeksforgeeks.org/machine-learning/weight-initialization-techniques-for-deep-neural-networks/) on GeeksforGeeks
  - [Xavier initialization](https://www.geeksforgeeks.org/deep-learning/xavier-initialization/) on GeeksforGeeks
  - [Kaiming Initialization in Deep Learning](https://www.geeksforgeeks.org/deep-learning/kaiming-initialization-in-deep-learning/) on GeeksforGeeks
- Activation
  - [The Importance and Reasoning behind Activation Functions](https://towardsdatascience.com/the-importance-and-reasoning-behind-activation-functions-4dc00e74db41/) by _Zach Brodtman_
  - [Neural networks: Multi-class classification](https://developers.google.com/machine-learning/crash-course/neural-networks/multi-class) on Google Developers
  - [Derivative of the Softmax Function and the Categorical Cross-Entropy Loss](https://medium.com/data-science/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1) by _Thomas Kurbiel_
  - [how can i take the derivative of the softmax output in back-prop](https://stackoverflow.com/questions/57631507/how-can-i-take-the-derivative-of-the-softmax-output-in-back-prop) on StackOverflow
  - [Softmax and Backpropagation](https://medium.com/@jsilvawasd/softmax-and-backpropagation-625c0c1f8241) by _Jsilvawasd_
  - [How to avoid numerical overflow in Sigmoid function: Numerically stable sigmoid function](https://shaktiwadekar.medium.com/how-to-avoid-numerical-overflow-in-sigmoid-function-numerically-stable-sigmoid-function-5298b14720f6) by _Shakti Wadekar_
- Cost
  - [Undestanding Cost Functions in Machine Learning: Types and Applications](https://medium.com/@anishnama20/understanding-cost-functions-in-machine-learning-types-and-applications-cd7d8cc4b47d) by _Anishnama_
- Training
  - [A (Very Short) Visual Introduction to Learning Rate Schedulers (With Code)](https://medium.com/@theom/a-very-short-visual-introduction-to-learning-rate-schedulers-with-code-189eddffdb00) by _Théo Martin_
  - [What is Gradient Clipping?](https://medium.com/data-science/what-is-gradient-clipping-b8e815cdfb48) by _Wanshun Wong_
- Metrics
  - [Classification: Accuracy, recall, precision, and related metrics](https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall) on Google Developers
  - [A Comprehensive Overview of Regression Evaluation Metrics](https://developer.nvidia.com/blog/a-comprehensive-overview-of-regression-evaluation-metrics/) on Nvidia Developer
- Other
  - [Packaging Python Project](https://packaging.python.org/en/latest/tutorials/packaging-projects) on Python Packaging User Guide

## Implementations

- [**MNIST Model**](https://github.com/sudthebud/ML-Model_MNIST)
  - Model trained to classify images in the MNIST dataset, a basic dataset of images of handwritten numerals
