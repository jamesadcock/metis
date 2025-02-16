import { logLoss, sigmoid } from "../functions/functions";
import { Matrix } from "../functions/matrix";

export class SingleLayerPerceptron {
  // Classify the input features using the provided weights
  public classify(features: Matrix, weights: Matrix, encoded = false): Matrix {
    // Add bias to the features
    const bias = Array.from({ length: features.rows }, () => 1);
    const featuresWithBias = features.addColumn(bias, 0);

    // Perform forward pass and return the result
    const output = this.forward(featuresWithBias, weights);

    // If encoded, return the index of the maximum value
    if (encoded) {
      return output.argMax();
    }

    // Otherwise, round the output values
    return output.applyFunction((x) => Math.round(x));
  }

  // Train the perceptron using the provided features and labels
  public train(
    features: Matrix,
    labels: Matrix,
    learningRate: number,
    epochs: number,
    showLoss = false,
  ): { weights: Matrix; loss: number } {
    // Add bias to the features
    const bias = Array.from({ length: features.rows }, () => 1);
    const featuresWithBias = features.addColumn(bias, 0);

    // Initialize weights with zeros
    let weights = new Matrix(
      Array.from({ length: featuresWithBias.columns }, () =>
        Array.from({ length: labels.columns }, () => 0),
      ),
    );

    // Training loop
    for (let epoch = 0; epoch < epochs; epoch++) {
      // Optionally show loss
      if (showLoss) {
        this.logLoss(featuresWithBias, labels, weights, epoch);
      }

      // Update weights using gradient descent
      const gradients = this.computeGradients(
        featuresWithBias,
        labels,
        weights,
      );
      weights = weights.subtractMatrices(gradients.multiply(learningRate));
    }

    // Return the final weights and loss
    const finalLoss = logLoss(labels, this.forward(featuresWithBias, weights));
    return { weights, loss: finalLoss };
  }

  // Perform the forward pass
  private forward(features: Matrix, weights: Matrix): Matrix {
    return features.multiplyMatrices(weights).applyFunction(sigmoid);
  }

  // Compute the gradients for weight update
  private computeGradients(
    features: Matrix,
    labels: Matrix,
    weights: Matrix,
  ): Matrix {
    const predictions = this.forward(features, weights);
    const predictionErrors = predictions.subtractMatrices(labels);
    const featuresTransposed = features.transpose();
    return featuresTransposed
      .multiplyMatrices(predictionErrors)
      .divide(features.rows);
  }

  // Log the loss at each iteration
  private logLoss(
    features: Matrix,
    labels: Matrix,
    weights: Matrix,
    iteration: number,
  ): void {
    const loss = logLoss(labels, this.forward(features, weights));
    console.log(`Iteration ${iteration} => Loss: ${loss}`);
  }
}
