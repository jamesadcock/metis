import { Matrix } from "../functions/matrix";

export interface TrainingData {
  trainingFeatureBatches: Matrix[];
  trainingLabelBatches: Matrix[];
  testingFeatures?: Matrix;
  testingLabels?: Matrix;
  validationFeatures?: Matrix;
  validationLabels?: Matrix;
  batchSize: number;
  lastBatchSize: number;
  numberOfBatches: number;
  unbatchedTrainingFeatures: Matrix;
  unbatchedTrainingLabels: Matrix;
}

export interface TrainingProps {
  trainingFeatureBatches: Matrix[];
  trainingLabelBatches: Matrix[];
  numberOfHiddenNodes: number;
  learningRate: number;
  epochs: number;
  report: boolean | undefined;
  testingFeatures: Matrix;
  testingLabels: Matrix;
  unbatchedTrainingFeatures: Matrix;
  unbatchedTrainingLabels: Matrix;
}
