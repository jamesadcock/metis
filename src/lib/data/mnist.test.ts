import { roundNumber } from "../../test/test-utilitity";
import { Mnist } from "./mnist";

// tests disabled because they take too long to run
// enable when working on the mnist class
describe.skip("mnist", () => {
  it("should load training data into a single batch when no batch size provided", () => {
    const mnist = new Mnist();
    const {
      trainingFeatureBatches: trainingFeatures,
      trainingLabelBatches: trainingLabels,
      batchSize,
      lastBatchSize,
      numberOfBatches,
      unbatchedTrainingFeatures: unbatchedFeatures,
      unbatchedTrainingLabels: unbatchedLabels,
      testingFeatures,
      testingLabels,
      validationFeatures,
      validationLabels,
    } = mnist.load(
      "test-data/mnist/train-images-idx3-ubyte",
      "test-data/mnist/train-labels-idx1-ubyte",
      "test-data/mnist/t10k-images-idx3-ubyte",
      "test-data/mnist/t10k-labels-idx1-ubyte",
    );

    expect(trainingFeatures[0].columns).toBe(28 * 28);
    expect(trainingFeatures[0].rows).toBe(60000);
    expect(trainingLabels[0].columns).toBe(10);
    expect(trainingLabels[0].rows).toBe(60000);
    expect(batchSize).toBe(0);
    expect(lastBatchSize).toBe(60000);
    expect(numberOfBatches).toBe(1);
    expect(unbatchedFeatures.columns).toBe(28 * 28);
    expect(unbatchedFeatures.rows).toBe(60000);
    expect(unbatchedLabels.columns).toBe(10);
    expect(unbatchedLabels.rows).toBe(60000);
    expect(testingFeatures.columns).toBe(28 * 28);
    expect(testingFeatures.rows).toBe(5000);
    expect(testingLabels.columns).toBe(1);
    expect(testingLabels.rows).toBe(5000);
    expect(validationFeatures.columns).toBe(28 * 28);
    expect(validationFeatures.rows).toBe(5000);
    expect(validationLabels.columns).toBe(1);
    expect(validationLabels.rows).toBe(5000);
  });

  it("should load training data and batch", () => {
    const mnist = new Mnist();
    const {
      trainingFeatureBatches: trainingFeatures,
      trainingLabelBatches: trainingLabels,
      batchSize,
      lastBatchSize,
      numberOfBatches,
      unbatchedTrainingFeatures: unbatchedFeatures,
      unbatchedTrainingLabels: unbatchedLabels,
      testingFeatures,
      testingLabels,
      validationFeatures,
      validationLabels,
    } = mnist.load(
      "test-data/mnist/train-images-idx3-ubyte",
      "test-data/mnist/train-labels-idx1-ubyte",
      "test-data/mnist/t10k-images-idx3-ubyte",
      "test-data/mnist/t10k-labels-idx1-ubyte",
      600,
    );

    expect(trainingFeatures[0].columns).toBe(28 * 28);
    expect(trainingFeatures[0].rows).toBe(600);
    expect(trainingLabels[0].columns).toBe(10);
    expect(trainingLabels[0].rows).toBe(600);
    expect(batchSize).toBe(600);
    expect(lastBatchSize).toBe(600);
    expect(numberOfBatches).toBe(100);
    expect(unbatchedFeatures.columns).toBe(28 * 28);
    expect(unbatchedFeatures.rows).toBe(60000);
    expect(unbatchedLabels.columns).toBe(10);
    expect(unbatchedLabels.rows).toBe(60000);
    expect(testingFeatures.columns).toBe(28 * 28);
    expect(testingFeatures.rows).toBe(5000);
    expect(testingLabels.columns).toBe(1);
    expect(testingLabels.rows).toBe(5000);
    expect(validationFeatures.columns).toBe(28 * 28);
    expect(validationFeatures.rows).toBe(5000);
    expect(validationLabels.columns).toBe(1);
    expect(validationLabels.rows).toBe(5000);
  });
});

describe("mnist standardize", () => {
  it("should standardize", () => {
    const mnist = new testMnist();
    const trainingSet = [[1, 2, 3]];
    const testSet = [[1, 2, 3]];
    const { trainingSetStandardized, testSetStandardized } = mnist.standardize(
      trainingSet,
      testSet,
    );
    expect(roundNumber(trainingSetStandardized[0][0], 3)).toEqual(-1.225);
    expect(roundNumber(trainingSetStandardized[0][1], 3)).toEqual(0);
    expect(roundNumber(trainingSetStandardized[0][2], 3)).toEqual(1.225);

    expect(roundNumber(testSetStandardized[0][0], 3)).toEqual(-1.225);
    expect(roundNumber(testSetStandardized[0][1], 3)).toEqual(0);
    expect(roundNumber(testSetStandardized[0][2], 3)).toEqual(1.225);
  });
});

class testMnist extends Mnist {
  public standardize(trainingSet: number[][], testSet: number[][]) {
    return super.standardize(trainingSet, testSet);
  }
}
