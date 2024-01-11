import { singleLayerPerceptron } from "./single-layer-perceptron";

describe("single-layer-perceptron", () => {
  it("returns 1", () => {
    const result = singleLayerPerceptron();
    expect(result).toEqual(1);
  });
});
