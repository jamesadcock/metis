import { Mnist } from "./data/mnist";
import { NeuralNetwork } from "./neural-network/neural-network";
import * as fs from "fs";

export const train = () => {
  const mnist = new Mnist();
  const {
    trainingFeatureBatches,
    trainingLabelBatches,
    unbatchedTrainingFeatures,
    unbatchedTrainingLabels,
    testingFeatures,
    testingLabels,
  } = mnist.load(
    "test-data/mnist/train-images-idx3-ubyte",
    "test-data/mnist/train-labels-idx1-ubyte",
    "test-data/mnist/t10k-images-idx3-ubyte",
    "test-data/mnist/t10k-labels-idx1-ubyte",
    1000
  );

  const trainingProps = {
    trainingFeatureBatches,
    trainingLabelBatches,
    numberOfHiddenNodes: 25,
    learningRate: 0.8,
    epochs: 100,
    report: true,
    testingFeatures,
    testingLabels,
    unbatchedTrainingFeatures,
    unbatchedTrainingLabels,
  };

  const neuralNet = new NeuralNetwork();
  const { weights1, weights2 } = neuralNet.train(trainingProps);

  // serialize the weights
  const weights1String = JSON.stringify(weights1);
  const weights2String = JSON.stringify(weights2);
  // save the weights to a file using fs
  fs.writeFileSync("weights1.json", weights1String);
  fs.writeFileSync("weights2.json", weights2String);
  return { weights1, weights2 };
};

train();
