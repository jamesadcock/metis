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
