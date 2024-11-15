import { Data } from "../data/data";
import { SingleLayerPerceptron } from "./single-layer-perceptron";
describe("iris", () => {
  it("should reduce the loss", () => {
    const trainingData = Data.loadTraining("test-data/iris-training.csv");
    const perceptron = new SingleLayerPerceptron();
    const result1 = perceptron.train(
      trainingData.features,
      trainingData.labels,
      0.001,
      10
    );
    const result2 = perceptron.train(
      trainingData.features,
      trainingData.labels,
      0.001,
      100
    );
    const result3 = perceptron.train(
      trainingData.features,
      trainingData.labels,
      0.001,
      1000
    );
    expect(result1.loss).toBeGreaterThan(result2.loss);
    expect(result2.loss).toBeGreaterThan(result3.loss);
  });
  it("should correctly predict", () => {
    const trainingData = Data.loadTraining("test-data/iris-training.csv");
    const perceptron = new SingleLayerPerceptron();
    const result = perceptron.train(
      trainingData.features,
      trainingData.labels,
      0.001,
      10000
    );
    const classificationData = Data.load("test-data/iris-test.csv");
    const results = perceptron.classify(
      classificationData.data,
      result.weights
    );

    expect(results.get()[0][0]).toEqual(0);
    expect(results.get()[1][0]).toEqual(1);
    expect(results.get()[2][0]).toEqual(0);
  });
});
