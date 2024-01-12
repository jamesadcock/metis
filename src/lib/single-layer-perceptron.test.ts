import {
  SingleLayerPerceptron,
  SingleLayerPerceptronProps,
} from "./single-layer-perceptron";

const props: SingleLayerPerceptronProps = {
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
      perceptron.predictFeature([0.1, 0.5, 0.2]);
    }).toThrow(new Error("Model is not trained yet"));
  });

  it("predicts the correct value", () => {
    const perceptron = new SingleLayerPerceptron(props);
    perceptron.train();
    expect(perceptron.predictFeature([0.1, 0.5, 0.2])).toBe(0);
    expect(perceptron.predictFeature([0.2, 0.3, 0.1])).toBe(1);
    expect(perceptron.predictFeature([0.7, 0.4, 0.2])).toBe(0);
    expect(perceptron.predictFeature([0.1, 0.4, 0.3])).toBe(1);
  });
});

describe("iris data test", () => {
  const irisTrainingData = {
    features: [
      {
        params: [5, 3, 1.6, 0.2],
        target: 0,
      },
      {
        params: [5, 3.4, 1.6, 0.4],
        target: 0,
      },
      {
        params: [5.2, 3.5, 1.5, 0.2],
        target: 0,
      },
      {
        params: [5.2, 3.4, 1.4, 0.2],
        target: 0,
      },
      {
        params: [4.7, 3.2, 1.6, 0.2],
        target: 0,
      },
      {
        params: [4.8, 3.1, 1.6, 0.2],
        target: 0,
      },
      {
        params: [5.4, 3.4, 1.5, 0.4],
        target: 0,
      },
      {
        params: [5.2, 4.1, 1.5, 0.1],
        target: 0,
      },
      {
        params: [5.5, 4.2, 1.4, 0.2],
        target: 0,
      },
      {
        params: [4.9, 3.1, 1.5, 0.1],
        target: 0,
      },
      {
        params: [5, 3.2, 1.2, 0.2],
        target: 0,
      },
      {
        params: [5.5, 3.5, 1.3, 0.2],
        target: 0,
      },
      {
        params: [4.9, 3.1, 1.5, 0.1],
        target: 0,
      },
      {
        params: [4.4, 3, 1.3, 0.2],
        target: 0,
      },
      {
        params: [5.1, 3.4, 1.5, 0.2],
        target: 0,
      },
      {
        params: [5, 3.5, 1.3, 0.3],
        target: 0,
      },
      {
        params: [4.5, 2.3, 1.3, 0.3],
        target: 0,
      },
      {
        params: [4.4, 3.2, 1.3, 0.2],
        target: 0,
      },
      {
        params: [5, 3.5, 1.6, 0.6],
        target: 0,
      },
      {
        params: [5.1, 3.8, 1.9, 0.4],
        target: 0,
      },
      {
        params: [4.8, 3, 1.4, 0.3],
        target: 0,
      },
      {
        params: [5.1, 3.8, 1.6, 0.2],
        target: 0,
      },
      {
        params: [4.6, 3.2, 1.4, 0.2],
        target: 0,
      },
      {
        params: [5.3, 3.7, 1.5, 0.2],
        target: 0,
      },
      {
        params: [5, 3.3, 1.4, 0.2],
        target: 0,
      },
      {
        params: [6.6, 3, 4.4, 1.4],
        target: 1,
      },
      {
        params: [6.8, 2.8, 4.8, 1.4],
        target: 1,
      },
      {
        params: [6.7, 3, 5, 1.7],
        target: 1,
      },
      {
        params: [6, 2.9, 4.5, 1.5],
        target: 1,
      },
      {
        params: [5.7, 2.6, 3.5, 1],
        target: 1,
      },
      {
        params: [5.5, 2.4, 3.8, 1.1],
        target: 1,
      },
      {
        params: [5.5, 2.4, 3.7, 1],
        target: 1,
      },
      {
        params: [5.8, 2.7, 3.9, 1.2],
        target: 1,
      },
      {
        params: [6, 2.7, 5.1, 1.6],
        target: 1,
      },
      {
        params: [5.4, 3, 4.5, 1.5],
        target: 1,
      },
      {
        params: [6, 3.4, 4.5, 1.6],
        target: 1,
      },
      {
        params: [6.7, 3.1, 4.7, 1.5],
        target: 1,
      },
      {
        params: [6.3, 2.3, 4.4, 1.3],
        target: 1,
      },
      {
        params: [5.6, 3, 4.1, 1.3],
        target: 1,
      },
      {
        params: [5.5, 2.5, 4, 1.3],
        target: 1,
      },
      {
        params: [5.5, 2.6, 4.4, 1.2],
        target: 1,
      },
      {
        params: [6.1, 3, 4.6, 1.4],
        target: 1,
      },
      {
        params: [5.8, 2.6, 4, 1.2],
        target: 1,
      },
      {
        params: [5, 2.3, 3.3, 1],
        target: 1,
      },
      {
        params: [5.6, 2.7, 4.2, 1.3],
        target: 1,
      },
      {
        params: [5.7, 3, 4.2, 1.2],
        target: 1,
      },
      {
        params: [5.7, 2.9, 4.2, 1.3],
        target: 1,
      },
      {
        params: [6.2, 2.9, 4.3, 1.3],
        target: 1,
      },
      {
        params: [5.1, 2.5, 3, 1.1],
        target: 1,
      },
      {
        params: [5.7, 2.8, 4.1, 1.3],
        target: 1,
      },
    ],
    learningRate: 0.5,
    epochs: 10,
  };

  const perceptron = new SingleLayerPerceptron(irisTrainingData);
  beforeAll(() => {
    perceptron.train();
  });

  it("predicts the correct 0 value", () => {
    expect(true).toBe(true);
  });

  it("predicts the correct 0 value", () => {
    expect(perceptron.predictFeature([5.1, 3.5, 1.4, 0.2])).toBe(0);
    expect(perceptron.predictFeature([4.9, 3, 1.4, 0.2])).toBe(0);
    expect(perceptron.predictFeature([5.4, 3.9, 1.7, 0.4])).toBe(0);
    expect(perceptron.predictFeature([5, 3.4, 1.5, 0.2])).toBe(0);
    expect(perceptron.predictFeature([5.1, 3.7, 1.5, 0.4])).toBe(0);
  });

  it("predicts the correct 1 value", () => {
    expect(perceptron.predictFeature([6.4, 3.2, 4.5, 1.5])).toBe(1);
    expect(perceptron.predictFeature([6.9, 3.1, 4.9, 1.5])).toBe(1);
    expect(perceptron.predictFeature([6.7, 3.1, 4.4, 1.4])).toBe(1);
    expect(perceptron.predictFeature([5.6, 3, 4.5, 1.5])).toBe(1);
    expect(perceptron.predictFeature([6.4, 2.9, 4.3, 1.3])).toBe(1);
  });
});
