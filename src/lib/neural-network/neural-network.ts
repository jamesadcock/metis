import {
  logLoss,
  sigmoid,
  sigmoidGradient,
  softmax,
} from "../functions/functions";
import { Matrix } from "../functions/matrix";

export class NeuralNetwork {
  public classify(
    features: Matrix,
    weights1: Matrix,
    weights2: Matrix,
    encoded = false
  ) {
    if (encoded) {
      return this.forward(features, weights1, weights2).predictions.argMax();
    }

    return this.forward(features, weights1, weights2).predictions.applyFunction(
      (x) => Math.round(x)
    );
  }

  // public train(
  //   features: Matrix,
  //   labels: Matrix,
  //   numberOfHiddenNodes: number,
  //   learningRate: number,
  //   epochs: number,
  //   showLoss = false
  // ) {
  //   const bias = Array.from({ length: features.rows }, () => 1);
  //   const featuresWithBias = features.addColumn(bias);

  //   let weights = new Matrix(
  //     Array.from({ length: featuresWithBias.columns }, () =>
  //       Array.from({ length: labels.columns }, () => 0)
  //     )
  //   );

  //   for (let i = 0; i < epochs; i++) {
  //     if (showLoss) {
  //       this.showLoss(featuresWithBias, labels, weights, i);
  //     }
  //     weights = weights.subtractMatrices(
  //       this.gradient(featuresWithBias, labels, weights).multiply(learningRate)
  //     );
  //   }
  //   return {
  //     weights: weights,
  //     loss: logLoss(labels, this.forward(featuresWithBias, weights)),
  //   };
  // }

  protected backPropagation(
    features: Matrix,
    labels: Matrix,
    predictions: Matrix,
    firstLayerOutput: Matrix,
    weight2: Matrix
  ) {
    const weight2Gradient = this.prependBias(firstLayerOutput)
      .transpose()
      .multiplyMatrices(predictions.subtractMatrices(labels))
      .divide(features.rows);

    const weight1Gradient = this.prependBias(features)
      .transpose()
      .multiplyMatrices(
        predictions
          .subtractMatrices(labels)
          .multiplyMatrices(weight2.removeRow(0).transpose())
          .elementWiseMultiplication(
            firstLayerOutput.applyFunction(sigmoidGradient)
          )
      )
      .divide(features.rows);

    return { weight2Gradient, weight1Gradient };
  }

  private prependBias(features: Matrix): Matrix {
    const bias = Array.from({ length: features.rows }, () => 1);
    const featuresWithBias = features.addColumn(bias, 0);
    return featuresWithBias;
  }

  protected forward(features: Matrix, weight1: Matrix, weight2: Matrix) {
    const firstLayerOutput = this.prependBias(features)
      .multiplyMatrices(weight1)
      .applyFunction(sigmoid); 

    const predictions =
      this.prependBias(firstLayerOutput).multiplyMatrices(weight2);
    return { predictions: softmax(predictions), firstLayerOutput };
  }

  private gradient(
    features: Matrix,
    labels: Matrix,
    weights1: Matrix,
    weights2: Matrix
  ) {
    const predictionErrors = this.forward(
      features,
      weights1,
      weights2
    ).predictions.subtractMatrices(labels);

    const featuresTransposed = features.transpose();
    return featuresTransposed
      .multiplyMatrices(predictionErrors)
      .divide(features.rows);
  }


  protected initializeWeights(
    nInputVariables: number,
    nHiddenNodes: number,
    nClasses: number
  ) {
    const w1Rows = nInputVariables + 1;
    const w1 = Matrix.random(w1Rows, nHiddenNodes).multiply(
      Math.sqrt(1 / w1Rows)
    );

    const w2Rows = nHiddenNodes + 1;
    const w2 = Matrix.random(w2Rows, nClasses).multiply(Math.sqrt(1 / w2Rows));

    return { w1, w2 };
  }

  private showLoss(
    features: Matrix,
    labels: Matrix,
    weights1: Matrix,
    weights2: Matrix,
    iteration: number
  ) {
    console.log(
      `Iterations ${iteration} => loss: ${logLoss(
        labels,
        this.forward(features, weights1, weights2).predictions
      )}`
    );
  }
}
