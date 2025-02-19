import { Mnist } from "./mnist";

describe("mnist load training data", () => {
  it("should load training data into a single batch when no batch size provided", () => {
    const mnist = new Mnist();
    const data = mnist.loadTrainingData(
      "test-data/mnist/train-images-idx3-ubyte",
      "test-data/mnist/train-labels-idx1-ubyte",
    );

    expect(data.features[0].columns).toBe(28 * 28);
    expect(data.features[0].rows).toBe(60000);
  });

  it("should load training data into multiple batches when batch size provided", () => {
    const mnist = new Mnist();
    const data = mnist.loadTrainingData(
      "test-data/mnist/train-images-idx3-ubyte",
      "test-data/mnist/train-labels-idx1-ubyte",
      100,
    );

    expect(data.features.length).toBe(600);
    expect(data.features[0].columns).toBe(28 * 28);
    expect(data.features[0].rows).toBe(100);
  });

  it("should throw error when file not found", () => {
    const mnist = new Mnist();
    expect(() =>
      mnist.loadTrainingData(
        "test-data/mnist/not-found",
        "test-data/mnist/not-found",
      ),
    ).toThrow("File not found: test-data/mnist/not-found");
  });
});

describe("test data", () => {
  it("should load test labels", () => {
    const mnist = new Mnist();
    const { test, validate } = mnist.loadTestAndValidationLabels(
      "test-data/mnist/t10k-labels-idx1-ubyte",
    );

    expect(test.columns).toBe(1);
    expect(test.rows).toBe(5000);

    expect(validate.columns).toBe(1);
    expect(validate.rows).toBe(5000);
  });

  it("should load test images", () => {
    const mnist = new Mnist();
    const { test, validate } = mnist.loadTestAndValidationImages(
      "test-data/mnist/t10k-images-idx3-ubyte",
    );

    expect(test.columns).toBe(28 * 28);
    expect(test.rows).toBe(5000);

    expect(validate.columns).toBe(28 * 28);
    expect(validate.rows).toBe(5000);
  });
});

describe("mnist one hot encode", () => {
  it("should one hot encode", () => {
    const mnist = new Mnist();
    const labels = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]];
    const oneHotEncoded = mnist.oneHotEncode(labels);

    expect(oneHotEncoded.get()).toEqual([
      [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    ]);
  });
});
