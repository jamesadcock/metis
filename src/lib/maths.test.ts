import { sigmoid } from "./maths";

describe("sigmoid", () => {
  it.each([
    [0, 0.5],
    [1, 0.7310585786],
    [-1, 0.2689414214],
  ])(
    "An input of %s results in an output of %s",
    (input: number, output: number) => {
      const result = sigmoid(input);
      expect(result.toFixed(10)).toEqual(output.toFixed(10));
    },
  );
});
