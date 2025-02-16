import { Matrix } from "../functions/matrix";
import { NeuralNetwork } from "./neural-network";

describe("backPropagation", () => {
  it("should return 0 for gradients when predictions are correct", () => {
    const neuralNetwork = new NeuralNetwork();
    const features = new Matrix([
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
    ]);
    const labels = new Matrix([[1], [1], [1]]);
    const predictions = new Matrix([[1], [1], [1]]);
    const firstLayerOutput = new Matrix([
      [0.5, 0.6],
      [0.7, 0.8],
      [0.9, 1.0],
    ]);
    const weight2 = new Matrix([[0.1], [0.2], [0.3]]);

    const result = neuralNetwork.backPropagation(
      features,
      labels,
      predictions,
      firstLayerOutput,
      weight2
    );

    expect(result.weight1Gradient.get()).toEqual([
      [0, 0],
      [0, 0],
      [0, 0],
      [0, 0],
    ]);
  });
  it("should return correct values for gradient 1 and 2 when predictions are not correct", () => {
    const neuralNetwork = new NeuralNetwork();
    const features = new Matrix([
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
    ]);
    const labels = new Matrix([[0.1], [0.2], [0.3]]);
    const predictions = new Matrix([[0.4], [0.5], [0.6]]);
    const firstLayerOutput = new Matrix([
      [0.5, 0.6],
      [0.7, 0.8],
      [0.9, 1.0],
    ]);
    const weight2 = new Matrix([[0.1], [0.2], [0.3]]);

    const result = neuralNetwork.backPropagation(
      features,
      labels,
      predictions,
      firstLayerOutput,
      weight2
    );

    const roundedWeight1Gradient = roundMatrix(result.weight1Gradient.get(), 3);

    expect(roundedWeight1Gradient).toEqual([
      [0.011, 0.012],
      [0.034, 0.026],
      [0.045, 0.038],
      [0.056, 0.05],
    ]);
  });
});

function roundMatrix(matrix: number[][], decimalPlaces: number): number[][] {
  return matrix.map((row) => {
    return row.map((num) => {
      const multiplier = Math.pow(10, decimalPlaces);
      return Math.round(num * multiplier) / multiplier;
    });
  });
}
