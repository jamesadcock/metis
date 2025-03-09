import { TrainingProps } from "../data/interfaces";
import {
  crossEntropyLoss,
  sigmoid,
  sigmoidGradient,
  softmax,
} from "../functions/functions";
import { Matrix } from "../functions/matrix";

export class NeuralNetwork {
  public classify(features: Matrix, weights1: Matrix, weights2: Matrix) {
    return this.forward(features, weights1, weights2).predictions.argMax();
  }

  public train(props: TrainingProps) {
    const {
      trainingFeatureBatches: featureBatches,
      trainingLabelBatches: labels,
      numberOfHiddenNodes,
      learningRate,
      epochs,
      report,
      testingFeatures,
      testingLabels,
      unbatchedTrainingFeatures: unbatchedFeatures,
      unbatchedTrainingLabels: unbatchedLabels,
    } = props;

    const nInputVariables = featureBatches[0].columns;
    const nClasses = labels[0].columns;

    let { w1, w2 } = this.initializeWeights(
      nInputVariables,
      numberOfHiddenNodes,
      nClasses,
    );

    for (let i = 0; i < epochs; i++) {
      for (let j = 0; j < featureBatches.length; j++) {
        const { predictions, firstLayerOutput } = this.forward(
          featureBatches[j],
          w1,
          w2,
        );

        const { weight2Gradient, weight1Gradient } = this.backPropagation(
          featureBatches[j],
          labels[j],
          predictions,
          firstLayerOutput,
          w2,
        );

        w2 = w2.subtractMatrices(weight2Gradient.multiply(learningRate));
        w1 = w1.subtractMatrices(weight1Gradient.multiply(learningRate));

        if (Number.isNaN(w2.get()[0][0]) || Number.isNaN(w1.get()[0][0])) {
          break;
        }
      }
      if (report) {
        this.report(
          testingFeatures,
          testingLabels,
          unbatchedFeatures,
          unbatchedLabels,
          w1,
          w2,
          i,
        );
      }
    }

    const { predictions } = this.forward(unbatchedFeatures, w1, w2);
    const loss = crossEntropyLoss(unbatchedLabels, predictions);

    return { weights1: w1, weights2: w2, loss };
  }

  protected backPropagation(
    features: Matrix,
    labels: Matrix,
    predictions: Matrix,
    firstLayerOutput: Matrix,
    weight2: Matrix,
  ) {
    const weight2Gradient = this.prependBias(firstLayerOutput)
      .transpose()
      .multiplyMatrices(predictions.subtractMatrices(labels))
      .divide(features.rows);

    const weight1Gradient = this.prependBias(features)
      .transpose()
      .multiplyMatrices(
        predictions
          .subtractMatrices(labels)
          .multiplyMatrices(weight2.removeRow(0).transpose())
          .elementWiseMultiplication(
            firstLayerOutput.applyFunction(sigmoidGradient),
          ),
      )
      .divide(features.rows);

    return { weight2Gradient, weight1Gradient };
  }

  private prependBias(features: Matrix): Matrix {
    const bias = Array.from({ length: features.rows }, () => 1);
    const featuresWithBias = features.addColumn(bias, 0);
    return featuresWithBias;
  }

  protected forward(features: Matrix, weight1: Matrix, weight2: Matrix) {
    const product = this.prependBias(features).multiplyMatrices(weight1);
    const firstLayerOutput = product.applyFunction(sigmoid);
    const predictions =
      this.prependBias(firstLayerOutput).multiplyMatrices(weight2);
    return { predictions: softmax(predictions), firstLayerOutput };
  }

  protected initializeWeights(
    nInputVariables: number,
    nHiddenNodes: number,
    nClasses: number,
  ) {
    const w1Rows = nInputVariables + 1;
    const w1 = Matrix.random(w1Rows, nHiddenNodes).multiply(
      Math.sqrt(1 / w1Rows),
    );

    const w2Rows = nHiddenNodes + 1;
    const w2 = Matrix.random(w2Rows, nClasses).multiply(Math.sqrt(1 / w2Rows));
    return { w1, w2 };
  }

  private report(
    testFeatures: Matrix,
    testLabels: Matrix,
    trainFeatures: Matrix,
    trainLabels: Matrix,
    w1: Matrix,
    w2: Matrix,
    epoch: number,
  ) {
    const { predictions } = this.forward(trainFeatures, w1, w2);
    const loss = crossEntropyLoss(trainLabels, predictions);
    const classifications = this.classify(testFeatures, w1, w2);
    const accuracy =
      classifications
        .subtractMatrices(testLabels)
        .applyFunction((input) => (input === 0 ? 1 : 0))
        .mean() * 100;
    console.log(
      `${epoch} > Loss: ${loss.toFixed(8)}, Accuracy: ${accuracy.toFixed(2)}%`,
    );
  }
}
