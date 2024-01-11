import {
  singleLayerPerceptron,
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
  epochs: 100,
};

describe("single-layer-perceptron", () => {
  it("reduces distance on each epoch", () => {
    const result1 = singleLayerPerceptron({ ...props, epochs: 10 });
    const result2 = singleLayerPerceptron({ ...props, epochs: 50 });
    const result3 = singleLayerPerceptron({ ...props, epochs: 100 });
    const result4 = singleLayerPerceptron({ ...props, epochs: 150 });
    const result5 = singleLayerPerceptron({ ...props, epochs: 200 });

    expect(result1.averageLoss).toBeGreaterThan(result2.averageLoss);
    expect(result2.averageLoss).toBeGreaterThan(result3.averageLoss);
    expect(result3.averageLoss).toBeGreaterThan(result4.averageLoss);
    expect(result4.averageLoss).toBeGreaterThan(result5.averageLoss);
  });

  it("returns weights and bias", () => {
    const result = singleLayerPerceptron(props);
    expect(result.bias).toBeDefined();
    expect(result.weights).toBeDefined();
  });
});
