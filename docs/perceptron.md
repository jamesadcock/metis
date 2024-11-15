# Perceptron

A single-layer perceptron is a type of artificial neural network and the simplest form of a feed forward network. Developed by Frank Rosenblatt in the late 1950s, it serves as a foundational concept in the field of neural networks and machine learning. Here's a breakdown of its main characteristics:

## Structure

A single-layer perceptron consists of a single layer of output nodes; the inputs are fed directly to the outputs via a series of weights. In essence, it maps input data into a desired output.

![Single Layer Perceptron](/docs/image/single-layer-perceptron.png)

## Input and Output

The perceptron receives multiple input signals, and each input is associated with a weight that modifies the strength of the signal. These inputs and weights are summed, and then the total is passed through an activation function to produce the output.

## Learning Process

The perceptron learns by adjusting the weights based on the errors in predictions. During training, it uses these errors to make small adjustments to the weights.

## Limitations

A single-layer perceptron is only capable of learning linearly separable patterns. This means they can only classify data that can be separated into classes with a linear boundary. They are not capable of solving problems where the data is not linearly separable, like the XOR problem.

## Uses

Despite its limitations, the single-layer perceptron is useful for simple binary classification tasks and serves as a stepping stone to more complex neural network architectures.

## Summary

In summary, the single-layer perceptron is a basic neural network model used for simple classification tasks, notable for its historical significance and simplicity, but limited by its inability to handle complex, non-linearly separable data.
