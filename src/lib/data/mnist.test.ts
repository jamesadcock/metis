import { Mnist } from "./mnist";

describe("mnist load images", () => {
  it("should load images", () => {
    const mnist = new Mnist();
    const images = mnist.loadImages("test-data/mnist/train-images-idx3-ubyte");

    expect(images.columns).toBe(28 * 28);
    expect(images.rows).toBe(60000);
  });

  it("should throw error when file not found", () => {
    const mnist = new Mnist();
    expect(() => mnist.loadImages("test-data/mnist/not-found")).toThrowError(
      "File not found: test-data/mnist/not-found"
    );
  });
});

describe("mnist load training labels", () => {
  it("should load labels", () => {
    const mnist = new Mnist();
    const labels = mnist.loadTrainingLabels(
      "test-data/mnist/train-labels-idx1-ubyte"
    );

    expect(labels.columns).toBe(10);
    expect(labels.rows).toBe(60000);
  });

  it("should throw error when file not found", () => {
    const mnist = new Mnist();
    expect(() =>
      mnist.loadTrainingLabels("test-data/mnist/not-found")
    ).toThrowError("File not found: test-data/mnist/not-found");
  });
});

describe("loadTestLabels", () => {
    it("should load test labels", () => {
        const mnist = new Mnist();
        const labels = mnist.loadTestLabels("test-data/mnist/t10k-labels-idx1-ubyte");

        console.log(labels.get());
        expect(labels.columns).toBe(1);
        expect(labels.rows).toBe(10000);
    });
});

describe("mnist one hot encode", () => {
  it("should one hot encode", () => {
    const mnist = new Mnist();
    const labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
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
