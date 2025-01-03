import { logLoss, sigmoid } from "../functions/functions";
import { Matrix } from "../functions/matrix";

export class NeuralNetwork {
  public classify(features: Matrix, weights: Matrix, encoded = false) {
    const bias = Array.from({ length: features.rows }, () => 1);
    const featuresWithBias = features.addColumn(bias);

    if (encoded) {
      return this.forward(featuresWithBias, weights).argMax();
    }

    return this.forward(featuresWithBias, weights).applyFunction((x) =>
      Math.round(x),
    );
  }

  public train(
    features: Matrix,
    labels: Matrix,
    learningRate: number,
    epochs: number,
    showLoss = false,
  ) {
    const bias = Array.from({ length: features.rows }, () => 1);
    const featuresWithBias = features.addColumn(bias);

    let weights = new Matrix(
      Array.from({ length: featuresWithBias.columns }, () =>
        Array.from({ length: labels.columns }, () => 0),
      ),
    );

    for (let i = 0; i < epochs; i++) {
      if (showLoss) {
        this.showLoss(featuresWithBias, labels, weights, i);
      }
      weights = weights.subtractMatrices(
        this.gradient(featuresWithBias, labels, weights).multiply(learningRate),
      );
    }
    return {
      weights: weights,
      loss: logLoss(labels, this.forward(featuresWithBias, weights)),
    };
  }

  private forward(features: Matrix, weights: Matrix) {
    return features.multiplyMatrices(weights).applyFunction(sigmoid);
  }

  private gradient(features: Matrix, labels: Matrix, weights: Matrix) {
    const predictionErrors = this.forward(features, weights).subtractMatrices(
      labels,
    );

    const featuresTransposed = features.transpose();
    return featuresTransposed
      .multiplyMatrices(predictionErrors)
      .divide(features.rows);
  }

  private showLoss(
    features: Matrix,
    labels: Matrix,
    weights: Matrix,
    iteration: number,
  ) {
    console.log(
      `Iterations ${iteration} => loss: ${logLoss(
        labels,
        this.forward(features, weights),
      )}`,
    );
  }
}
