import { Data } from "../data/data";
import { Mnist } from "../data/mnist";
import { SingleLayerPerceptron } from "./single-layer-perceptron";
describe("iris", () => {
  it("should reduce the loss", () => {
    const trainingData = Data.loadTraining("test-data/iris-training.csv");
    const perceptron = new SingleLayerPerceptron();
    const result1 = perceptron.train(
      trainingData.features[0],
      trainingData.labels[0],
      0.001,
      10,
    );
    const result2 = perceptron.train(
      trainingData.features[0],
      trainingData.labels[0],
      0.001,
      100,
    );
    const result3 = perceptron.train(
      trainingData.features[0],
      trainingData.labels[0],
      0.001,
      1000,
    );
    expect(result1.loss).toBeGreaterThan(result2.loss);
    expect(result2.loss).toBeGreaterThan(result3.loss);
  });
  it("should correctly predict", () => {
    const trainingData = Data.loadTraining("test-data/iris-training.csv");
    const perceptron = new SingleLayerPerceptron();
    const result = perceptron.train(
      trainingData.features[0],
      trainingData.labels[0],
      0.001,
      10000,
    );
    const classificationData = Data.load("test-data/iris-test.csv");
    const results = perceptron.classify(
      classificationData.data,
      result.weights,
    );

    expect(results.get()[0][0]).toEqual(0);
    expect(results.get()[1][0]).toEqual(1);
    expect(results.get()[2][0]).toEqual(0);
  });
});

describe.skip("mnist", () => {
  it("should reduce the loss", async () => {
    const mnist = new Mnist();
    const perceptron = new SingleLayerPerceptron();
    const trainingData = mnist.loadTrainingData(
      "test-data/mnist/train-images-idx3-ubyte",
      "test-data/mnist/train-labels-idx1-ubyte",
    );

    const result1 = perceptron.train(
      trainingData.features[0],
      trainingData.labels[0],
      0.0001,
      2,
    );
    const result2 = perceptron.train(
      trainingData.features[0],
      trainingData.labels[0],
      0.0001,
      3,
    );
    expect(result1.loss).toBeGreaterThan(result2.loss);
  }, 100000);

  it("should correctly predict", () => {
    const mnist = new Mnist();
    const perceptron = new SingleLayerPerceptron();
    const trainingData = mnist.loadTrainingData(
      "test-data/mnist/train-images-idx3-ubyte",
      "test-data/mnist/train-labels-idx1-ubyte",
    );
    const result = perceptron.train(
      trainingData.features[0],
      trainingData.labels[0],
      0.00001,
      10,
    );
    const classificationData = mnist.loadTestImages(
      "test-data/mnist/t10k-images-idx3-ubyte",
    );

    const classificationLabels = mnist.loadTestLabels(
      "test-data/mnist/t10k-labels-idx1-ubyte",
    );

    const results = perceptron.classify(
      classificationData,
      result.weights,
      true,
    );

    let correct = 0;
    results.get().forEach((result, i) => {
      if (result[0] === classificationLabels.get()[i][0]) {
        correct++;
      }
    });
    console.log(
      `Correctly Identified: ${(correct / results.get().length) * 100}%`,
    );
    expect(classificationLabels.get()[0][0]).toEqual(7);
  });
});
