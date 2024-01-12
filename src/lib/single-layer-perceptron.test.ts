import {
  SingleLayerPerceptron,
  SingleLayerPerceptronProps,
} from "./single-layer-perceptron";

const props: SingleLayerPerceptronProps = {
  bias: 0.5,
  features: [
    {
      params: [0.1, 0.5, 0.2],
      target: 0,
    },
    {
      params: [0.2, 0.3, 0.1],
      target: 1,
    },
    {
      params: [0.7, 0.4, 0.2],
      target: 0,
    },
    {
      params: [0.1, 0.4, 0.3],
      target: 1,
    },
  ],
  weights: [0.4, 0.2, 0.6],
  learningRate: 0.1,
  epochs: 10000,
};

describe("train", () => {
  it("reduces distance on each epoch", () => {
    const result1 = new SingleLayerPerceptron({ ...props, epochs: 10 }).train();
    const result2 = new SingleLayerPerceptron({ ...props, epochs: 50 }).train();
    const result3 = new SingleLayerPerceptron({
      ...props,
      epochs: 100,
    }).train();
    const result4 = new SingleLayerPerceptron({
      ...props,
      epochs: 150,
    }).train();
    const result5 = new SingleLayerPerceptron({
      ...props,
      epochs: 200,
    }).train();

    expect(result1.averageLoss).toBeGreaterThan(result2.averageLoss);
    expect(result2.averageLoss).toBeGreaterThan(result3.averageLoss);
    expect(result3.averageLoss).toBeGreaterThan(result4.averageLoss);
    expect(result4.averageLoss).toBeGreaterThan(result5.averageLoss);
  });

  it("returns weights and bias", () => {
    const result = new SingleLayerPerceptron(props).train();
    expect(result.bias).toBeDefined();
    expect(result.weights).toBeDefined();
  });
});

describe("predict", () => {
  it("throws an error if the perceptron is not trained", () => {
    const perceptron = new SingleLayerPerceptron(props);
    expect(() => {
      perceptron.predictFeature({ params: [0.1, 0.5, 0.2], target: 0 });
    }).toThrow(new Error("Model is not trained yet"));
  });

  // write a test that checks if the perceptron predicts the correct value
  // for the given feature
  it("predicts the correct value", () => {
    const perceptron = new SingleLayerPerceptron(props);
    perceptron.train();
    expect(
      perceptron.predictFeature({ params: [0.1, 0.5, 0.2], target: 0 }),
    ).toBe(0);
    expect(
      perceptron.predictFeature({ params: [0.2, 0.3, 0.1], target: 1 }),
    ).toBe(1);
    expect(
      perceptron.predictFeature({ params: [0.7, 0.4, 0.2], target: 0 }),
    ).toBe(0);
    expect(
      perceptron.predictFeature({ params: [0.1, 0.4, 0.3], target: 1 }),
    ).toBe(1);
  });
});
