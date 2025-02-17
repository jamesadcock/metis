import { Matrix } from "../functions/matrix";
import { NeuralNetwork } from "./neural-network";
import { mockMathRandom, roundMatrix } from "../../test/test-utilitity";

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
      weight2
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
      weight2
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

describe("forward", () => {
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
      3
    );

    expect(roundedFirstLayerOutput).toEqual([
      [0.993, 0.996],
      [1, 1],
      [1, 1],
    ]);
    expect(roundedPredictions).toEqual([[0.333], [0.334], [0.334]]);
    expect(result.predictions.sum()).toEqual(1);
  });
});

describe("InitializeWeights", () => {
  mockMathRandom(0.5);
  it("should return correct values for weight1 and weight2", () => {
    const neuralNetwork = new NeuralNetworkTestWrapper();
    const nInputVariables = 3;
    const nHiddenNodes = 2;
    const nClasses = 1;

    const { w1, w2 } = neuralNetwork.testInitializeWeights(
      nInputVariables,
      nHiddenNodes,
      nClasses
    );

    expect(w1.get()).toEqual([
      [0.25, 0.25],
      [0.25, 0.25],
      [0.25, 0.25],
      [0.25, 0.25],
    ]);

    expect(roundMatrix(w2.get(), 3)).toEqual([[0.289], [0.289], [0.289]]);
  });
});

class NeuralNetworkTestWrapper extends NeuralNetwork {
  public testBackPropagation(
    features: Matrix,
    labels: Matrix,
    predictions: Matrix,
    firstLayerOutput: Matrix,
    weight2: Matrix
  ) {
    return this.backPropagation(
      features,
      labels,
      predictions,
      firstLayerOutput,
      weight2
    );
  }

  public testForward(features: Matrix, weight1: Matrix, weight2: Matrix) {
    return this.forward(features, weight1, weight2);
  }

  public testInitializeWeights(
    nInputVariables: number,
    nHiddenNodes: number,
    nClasses: number
  ) {
    return this.initializeWeights(nInputVariables, nHiddenNodes, nClasses);
  }
}
