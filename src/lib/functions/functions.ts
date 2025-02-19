import { Matrix } from "./matrix";

export const sigmoid = (input: number): number => {
  return 1 / (1 + Math.exp(-input));
};

export const logLoss = (labels: Matrix, predictions: Matrix) => {
  const firstTerm = labels.elementWiseMultiplication(
    predictions.applyFunction(Math.log),
  );

  const secondTerm = predictions
    .applyFunction((x) => 1 - x)
    .applyFunction(Math.log)
    .elementWiseMultiplication(labels.applyFunction((x) => 1 - x));

  return firstTerm.addMatrices(secondTerm).multiply(-1).mean();
};

export const crossEntropyLoss = (labels: Matrix, predictions: Matrix) => {
  // Clip the predictions to avoid log(0)
  const epsilon = 1e-15;
  const clippedPredictions = predictions.applyFunction((x) =>
    Math.max(Math.min(x, 1 - epsilon), epsilon),
  );

  // Calculate cross-entropy loss
  return -labels
    .elementWiseMultiplication(clippedPredictions.applyFunction(Math.log))
    .mean();
};

/*
def loss(Y, y_hat):
    return -np.sum(Y * np.log(y_hat)) / Y.shape[0]
*/

export const softmax = (logits: Matrix): Matrix => {
  // Subtract the maximum logit value from each logit to prevent overflow
  const maxLogit = logits.matrixMax();

  return logits
    .subtract(maxLogit)
    .applyFunction(Math.exp)
    .divide(logits.subtract(maxLogit).applyFunction(Math.exp).sum());
};

export const sigmoidGradient = (sigmoid: number): number => {
  return sigmoid * (1 - sigmoid);
};

/*
1.817
1.822
1.822

5.46

0.333
0.334
0.334
  */
