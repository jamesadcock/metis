import { logLoss, sigmoid } from "./functions";
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
    }
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
