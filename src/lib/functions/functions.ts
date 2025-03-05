import { Matrix } from "./matrix";

export const sigmoid = (input: number): number => {
  return 1 / (1 + Math.exp(-input));
};

export const logLoss = (labels: Matrix, predictions: Matrix) => {
  const firstTerm = labels.elementWiseMultiplication(
    predictions.applyFunction(Math.log)
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
    Math.max(Math.min(x, 1 - epsilon), epsilon)
  );

  // Calculate cross-entropy loss
  return -labels
    .elementWiseMultiplication(clippedPredictions.applyFunction(Math.log))
    .mean();
};

export const softmax = (logits: Matrix): Matrix => {
  const exponentials = logits.applyFunction(Math.exp);
  const sumOfRows = exponentials.sumRows();
  return exponentials.divideRow(sumOfRows);
};

export const sigmoidGradient = (sigmoid: number): number => {
  return sigmoid * (1 - sigmoid);
};

export const calculateMean = (values: number[][]): number => {
  const flattenedArray = values.flat();
  const sum = flattenedArray.reduce((acc, val) => acc + val, 0);
  return sum / flattenedArray.length;
};

export const calculateStandardDeviation = (values: number[][]): number => {
  const flattenedArray = values.flat();
  const mean = calculateMean(values);
  const squaredDifferences = flattenedArray.map((value) =>
    Math.pow(value - mean, 2)
  );
  const sumOfSquaredDifferences = squaredDifferences.reduce(
    (acc, val) => acc + val,
    0
  );
  const variance = sumOfSquaredDifferences / flattenedArray.length;
  return Math.sqrt(variance);
};
