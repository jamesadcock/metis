import { crossEntropyLoss, mean, sigmoid } from "./functions";

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

describe("cross entropy loss", () => {
  it.each([[0, 0.68, 1.14]])(
    "A target of %s with a prediction of %s results in a loss of %s",
    (target: number, prediction: number, output: number) => {
      const result = crossEntropyLoss(target, prediction);
      expect(result.toFixed(2)).toEqual(output.toFixed(2));
    },
  );
});

describe("mean", () => {
  it.each([
    [[1, 9], 5],
    [[3, 3, 3], 3],
    [[0.49, 0.18, 0.55, 0.18], 0.35],
  ])(
    "The array of numbers %s should result in a mean of %s",
    (numbers: number[], output: number) => {
      const result = mean(numbers);
      expect(result.toFixed(2)).toEqual(output.toFixed(2));
    },
  );
});
