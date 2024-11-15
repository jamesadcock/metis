# Functions

## Sigmoid Function

A sigmoid function is a mathematical function having a characteristic "S"-shaped curve or sigmoid curve. It's widely used in machine learning, particularly in logistic regression and neural networks, due to its good properties for modeling probability and its smooth gradient. The most common example of a sigmoid function is the logistic function.

![Sigmoid Function Graph](/docs/image/sigmoid-function-graph.png)

Here are some key features of the sigmoid function:

**Formula:** The standard logistic sigmoid function is defined as:

![Sigmoid Function](/docs/image/sigmoid-function.png)

where e is Euler's number and x is the input to the function.

**Output Range:** The sigmoid function outputs a value between 0 and 1, which makes it especially useful for models where we need to predict probabilities.

**Non-linearity:** The function is non-linear, meaning it can help neural networks learn complex patterns.

**Smooth Gradient:** The function has a smooth gradient, avoiding sharp jumps in output values. This smoothness is beneficial during the optimization phase of training a neural network.

**Saturation and Vanishing Gradient Problem:** The sigmoid function has regions where the function flattens out near 0 and 1. In these regions, the gradient is very small, leading to the vanishing gradient problem during neural network training. This can slow down or even stop the network from further learning.

**Centered at 0.5:** The midpoint output of the sigmoid function is 0.5, not 0, which can sometimes be a disadvantage in neural networks as it leads to outputs that are not zero-centered.

In summary, the sigmoid function is a smooth, S-shaped function crucial in the early development of neural networks and logistic regression. Its ability to model probabilities and its smooth gradient are significant advantages, but it is less used in modern deep learning due to issues like the vanishing gradient problem.

## Log Loss

Log loss also known as cross entropy loss, also known, is a widely used loss function in machine learning and deep learning, particularly for classification problems. It measures the performance of a classification model whose output is a probability value between 0 and 1.

Cross entropy loss increases as the predicted probability diverges from the actual label, making it an effective measure for evaluating the accuracy of a classifier.

_Here's a more detailed look at cross entropy loss:_

Concept: The concept of cross entropy originates from information theory, where it is used to quantify the difference between two probability distributions.

Formula: For binary classification, the cross entropy loss can be defined as:

![Cross Entropy Loss](/docs/image/cross-entropy-loss.png)

Where:

- C: is the total number of classes.
- yi: is the true label for class (1 for the correct class, 0 otherwise).
- yi^: is the predicted probability for each class.

**Interpretation:** The loss is low (near zero) when the model's prediction is close to the actual label, and it's high when the prediction is far from the actual label.

**Usage in Training:** In machine learning, particularly in neural networks, cross entropy loss is used as a loss function to train the model. The goal is to minimize this loss during the training process.

**Advantages:** It works well with probabilities, making it suitable for models that output probabilities.
It is sensitive to the differences between the actual and predicted probabilities, which helps in learning more accurate models.

**Limitations:** For imbalanced datasets, standard cross entropy loss can bias the model towards the majority class. Modifications or alternative loss functions may be required in such cases.

In summary, cross entropy loss is an essential concept in machine learning for classification tasks, providing a measure of the difference between the actual and predicted probability distributions, and is used to guide the training of models towards more accurate predictions.
