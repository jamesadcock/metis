import { Matrix } from "../functions/matrix";

export class LinearRegression {
  public predict(features: Matrix, weights: Matrix) {
    return features.multiplyMatrices(weights);
  }

  public gradientDescent(features: Matrix, weights: Matrix, labels: Matrix) {
    const predictionErrors = this.predict(features, weights).subtractMatrices(
      labels
    );
    const featuresTransposed = features.transpose();
    return featuresTransposed
      .multiplyMatrices(predictionErrors)
      .divide(features.rows)
      .multiply(2);
  }

  private loss(features: Matrix, weights: Matrix, labels: Matrix) {
    return this.predict(features, weights)
      .subtractMatrices(labels)
      .squared()
      .mean();
  }

  public train(
    features: Matrix,
    labels: Matrix,
    learningRate: number,
    epochs: number
  ) {
    let weights = new Matrix(
      Array.from({ length: features.columns }, () => [0])
    );
    for (let i = 0; i < epochs; i++) {
      weights = weights.subtractMatrices(
        this.gradientDescent(features, weights, labels).multiply(learningRate)
      );
    }
    console.log(`squared loss=${this.loss(features, weights, labels)}`);
    return weights;
  }
}
