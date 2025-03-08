import { Matrix } from "../functions/matrix";
import { NeuralNetwork } from "./neural-network";
import {
  mockMathRandom,
  resetMockMathRandom,
  roundMatrix,
} from "../../test/test-utilitity";
import { Data } from "../data/data";
import { Mnist } from "../data/mnist";
import { TrainingProps } from "../data/interfaces";

describe("train", () => {
  let trainingProps: TrainingProps;

  beforeAll(() => {
    const {
      trainingFeatures,
      trainingLabels,
      unbatchedFeatures,
      unbatchedLabels,
    } = Data.loadTraining("test-data/iris-training.csv", 0, true);

    const { testFeatures: testingFeatures, testLabels: testingLabels } =
      Data.loadValidationAndTest("test-data/iris-test.csv");

    trainingProps = {
      trainingFeatureBatches: trainingFeatures,
      trainingLabelBatches: trainingLabels,
      numberOfHiddenNodes: 2,
      learningRate: 1,
      epochs: 10,
      showLoss: false,
      testingFeatures,
      testingLabels,
      unbatchedTrainingFeatures: unbatchedFeatures,
      unbatchedTrainingLabels: unbatchedLabels,
    };
  });

  it("should reduce loss", () => {
    const neuralNetwork = new NeuralNetwork();
    const result1 = neuralNetwork.train(trainingProps);
    const result2 = neuralNetwork.train({ ...trainingProps, epochs: 100 });
    const result3 = neuralNetwork.train({ ...trainingProps, epochs: 500 });

    expect(result1.loss).toBeGreaterThan(result2.loss);
    expect(result2.loss).toBeGreaterThan(result3.loss);
  });

  it("should correctly predict", () => {
    const neuralNetwork = new NeuralNetwork();
    const result = neuralNetwork.train({ ...trainingProps, epochs: 500 });

    const { validationFeatures, validationLabels } = Data.loadValidationAndTest(
      "test-data/iris-test.csv",
    );
    const results = neuralNetwork.classify(
      validationFeatures,
      result.weights1,
      result.weights2,
    );

    expect(results.get()[0][0]).toEqual(validationLabels.get()[0][0]);
    expect(results.get()[1][0]).toEqual(validationLabels.get()[1][0]);
  });
});

describe("backPropagation", () => {
  it("should return 0 for gradients when predictions are correct", () => {
    const neuralNetwork = new NeuralNetworkTestWrapper();
    const features = new Matrix([
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
    ]);
    const labels = new Matrix([[1], [1], [1]]);
    const predictions = new Matrix([[1], [1], [1]]);
    const firstLayerOutput = new Matrix([
      [0.5, 0.6],
      [0.7, 0.8],
      [0.9, 1.0],
    ]);
    const weight2 = new Matrix([[0.1], [0.2], [0.3]]);

    const result = neuralNetwork.testBackPropagation(
      features,
      labels,
      predictions,
      firstLayerOutput,
      weight2,
    );

    expect(result.weight1Gradient.get()).toEqual([
      [0, 0],
      [0, 0],
      [0, 0],
      [0, 0],
    ]);
  });
  it("should return correct values for gradient 1 and 2 when predictions are not correct", () => {
    const neuralNetwork = new NeuralNetworkTestWrapper();
    const features = new Matrix([
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
    ]);
    const labels = new Matrix([[0.1], [0.2], [0.3]]);
    const predictions = new Matrix([[0.4], [0.5], [0.6]]);
    const firstLayerOutput = new Matrix([
      [0.5, 0.6],
      [0.7, 0.8],
      [0.9, 1.0],
    ]);
    const weight2 = new Matrix([[0.1], [0.2], [0.3]]);

    const result = neuralNetwork.testBackPropagation(
      features,
      labels,
      predictions,
      firstLayerOutput,
      weight2,
    );

    const roundedWeight1Gradient = roundMatrix(result.weight1Gradient.get(), 3);

    expect(roundedWeight1Gradient).toEqual([
      [0.011, 0.012],
      [0.034, 0.026],
      [0.045, 0.038],
      [0.056, 0.05],
    ]);
  });
});

describe.skip("forward", () => {
  it("should return correct values for predictions and firstLayerOutput", () => {
    const neuralNetwork = new NeuralNetworkTestWrapper();
    // hidden nodes = 2
    // classes = 1

    const features = new Matrix([
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
    ]);

    const weight1 = new Matrix([
      [0.1, 0.2],
      [0.4, 0.5],
      [0.7, 0.8],
      [1.0, 1.1],
    ]);

    const weight2 = new Matrix([[0.1], [0.2], [0.3]]);

    /*
    First Layer Output
    features + bias = 3 x 4
    weight1 = 4 x 2
    result = 3 x 2

    [1, 1, 2, 3]     [0.1, 0.2]    [4.9, 5.6]
    [1, 4, 5, 6]  X  [0.4, 0.5]  = [11.2, 12.8]
    [1, 7, 8, 9]     [0.7, 0.8]    [17.5, 20]
                     [1.0, 1.1]  
                    
    [4.9, 6.6]                          [0.993, 0.996]              
    [1, 4, 5, 6]  X  1 / (1 + e^(-x)) = [1, 1]
    [1, 7, 8, 9]                        [1, 1]

    Predictions
    First Layer Output + bias = 3 x 3
    weight2 = 3 x 1
    result = 3 x 1

    [1, 0.993, 0.996]           [0.1]    [0.597]
    [1, 1    , 1    ]   X       [0.2]  = [0.6]
    [1, 1    , 1    ]           [0.3]    [0.6]

    [0.597]                     [0.333]
    [0.6]    X Softmax      =   [0.334]
    [0.6]                       [0.334]

    */

    const result = neuralNetwork.testForward(features, weight1, weight2);

    const roundedPredictions = roundMatrix(result.predictions.get(), 3);
    const roundedFirstLayerOutput = roundMatrix(
      result.firstLayerOutput.get(),
      3,
    );

    expect(roundedFirstLayerOutput).toEqual([
      [0.993, 0.996],
      [1, 1],
      [1, 1],
    ]);
    expect(roundedPredictions).toEqual([[0.333], [0.334], [0.334]]);
  });
});

describe("InitializeWeights", () => {
  beforeEach(() => {
    mockMathRandom(1);
  });

  afterEach(() => {
    resetMockMathRandom();
  });
  it("should return correct values for weight1 and weight2", () => {
    const neuralNetwork = new NeuralNetworkTestWrapper();
    const nInputVariables = 3;
    const nHiddenNodes = 2;
    const nClasses = 1;

    const { w1, w2 } = neuralNetwork.testInitializeWeights(
      nInputVariables,
      nHiddenNodes,
      nClasses,
    );

    expect(w1.get()).toEqual([
      [0.5, 0.5],
      [0.5, 0.5],
      [0.5, 0.5],
      [0.5, 0.5],
    ]);

    expect(roundMatrix(w2.get(), 3)).toEqual([[0.577], [0.577], [0.577]]);
  });
});

class NeuralNetworkTestWrapper extends NeuralNetwork {
  public testBackPropagation(
    features: Matrix,
    labels: Matrix,
    predictions: Matrix,
    firstLayerOutput: Matrix,
    weight2: Matrix,
  ) {
    return this.backPropagation(
      features,
      labels,
      predictions,
      firstLayerOutput,
      weight2,
    );
  }

  public testForward(features: Matrix, weight1: Matrix, weight2: Matrix) {
    return this.forward(features, weight1, weight2);
  }

  public testInitializeWeights(
    nInputVariables: number,
    nHiddenNodes: number,
    nClasses: number,
  ) {
    return this.initializeWeights(nInputVariables, nHiddenNodes, nClasses);
  }
}

describe.skip("mnist", () => {
  let trainingProps: TrainingProps;
  let validationData;

  beforeAll(() => {
    const mnist = new Mnist();
    const {
      trainingFeatures,
      trainingLabels,
      unbatchedFeatures,
      unbatchedLabels,
      testingFeatures,
      testingLabels,
      validationFeatures,
      validationLabels,
    } = mnist.load(
      "test-data/mnist/train-images-idx3-ubyte",
      "test-data/mnist/train-labels-idx1-ubyte",
      "test-data/mnist/t10k-images-idx3-ubyte",
      "test-data/mnist/t10k-labels-idx1-ubyte",
      1000,
    );

    trainingProps = {
      trainingFeatureBatches: trainingFeatures,
      trainingLabelBatches: trainingLabels,
      numberOfHiddenNodes: 25,
      learningRate: 0.8,
      epochs: 25,
      showLoss: true,
      testingFeatures,
      testingLabels,
      unbatchedTrainingFeatures: unbatchedFeatures,
      unbatchedTrainingLabels: unbatchedLabels,
    };

    validationData = {
      validationFeatures,
      validationLabels,
    };
  });

  it.only("should reduce the loss", async () => {
    const neuralNet = new NeuralNetwork();
    const { loss: loss1 } = neuralNet.train({ ...trainingProps, epochs: 1 });
    const { loss: loss2 } = neuralNet.train({ ...trainingProps, epochs: 2 });
    const { loss: loss3 } = neuralNet.train({ ...trainingProps, epochs: 3 });

    expect(loss1).toBeGreaterThan(loss2);
    expect(loss2).toBeGreaterThan(loss3);
  }, 100000);

  it("should correctly predict", async () => {
    const neuralNet = new NeuralNetwork();
    const { weights1, weights2 } = neuralNet.train({
      ...trainingProps,
      epochs: 5,
    });
    const result = neuralNet.classify(
      validationData.validationFeatures,
      weights1,
      weights2,
    );
    expect(result.get()[0][0]).toEqual(
      validationData.validationLabels.get()[0][0],
    );
    expect(result.get()[1][0]).toEqual(
      validationData.validationLabels.get()[1][0],
    );
    expect(result.get()[2][0]).toEqual(
      validationData.validationLabels.get()[2][0],
    );
  }, 100000);
});
