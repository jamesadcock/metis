import { roundMatrix, roundNumber } from "../../test/test-utilitity";
import {
  calculateMean,
  calculateStandardDeviation,
  crossEntropyLoss,
  logLoss,
  sigmoid,
  sigmoidGradient,
  softmax,
} from "./functions";
import { Matrix } from "./matrix";

describe("sigmoid", () => {
  it.each([
    [0, 0.5],
    [1, 0.7310585786],
    [-1, 0.2689414214],
  ])(
    "An input of %s results in an output of %s",
    (input: number, output: number) => {
      const result = sigmoid(input);
      expect(result.toFixed(2)).toEqual(output.toFixed(2));
    },
  );
});

describe("log loss", () => {
  it("should return the correct loss for single value", () => {
    const targets = new Matrix([[1]]);
    const predictions = new Matrix([[0.9]]);
    const result = logLoss(targets, predictions);
    expect(result).toEqual(0.10536051565782628);
  });

  it("should return the correct loss for multiple values", () => {
    const targets = new Matrix([[1], [0]]);
    const predictions = new Matrix([[0.9], [0.4]]);
    const result = logLoss(targets, predictions);
    expect(result).toEqual(0.30809306971190853);
  });
});

describe("softmax", () => {
  it("should return the correct softmax for single value", () => {
    const logits = new Matrix([[1]]);
    const result = softmax(logits);
    expect(result.get()).toEqual([[1]]);
  });

  it("should return the correct softmax for multiple values", () => {
    const logits = new Matrix([
      [2.0, 1.0, 0.1],
      [0.5, 1.5, 2.5],
    ]);

    const result = softmax(logits);
    const roundedResult = roundMatrix(result.get(), 3);

    expect(roundedResult).toEqual([
      [0.659, 0.242, 0.099],
      [0.09, 0.245, 0.665],
    ]);

    const sum = result.sum();
    expect(sum).toEqual(2);
  });
});

describe("cross entropy loss", () => {
  it("should return the correct loss for multiple values", () => {
    const targets = new Matrix([[0], [1], [0]]);
    const predictions = new Matrix([[0.2], [0.7], [0.1]]);
    const result = crossEntropyLoss(targets, predictions);
    const roundedResult = parseFloat(result.toFixed(3));
    expect(roundedResult).toEqual(0.119);
  });
});

describe("sigmoid gradient", () => {
  it("should return the correct sigmoid gradient for single value", () => {
    const sig = sigmoid(0.5);
    const result = sigmoidGradient(sig);
    const roundedResult = parseFloat(result.toFixed(3));
    expect(roundedResult).toEqual(0.235);
  });
});

describe("calculate mean", () => {
  it("should return the correct mean for multiple values", () => {
    const values = [
      [1, 2],
      [3, 4],
      [5, 6],
    ];
    const result = calculateMean(values);
    expect(result).toEqual(3.5);
  });
});

describe("calculate standard deviation", () => {
  it("should return the correct standard deviation for multiple values", () => {
    const values = [[1], [2], [3]];
    const result = calculateStandardDeviation(values);
    expect(roundNumber(result, 2)).toEqual(0.82);
  });
});
