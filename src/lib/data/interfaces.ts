import { Matrix } from "../functions/matrix";

export interface TrainingData {
  trainingFeatures: Matrix[];
  trainingLabels: Matrix[];
  testingFeatures?: Matrix;
  testingLabels?: Matrix;
  validationFeatures?: Matrix;
  validationLabels?: Matrix;
  batchSize: number;
  lastBatchSize: number;
  numberOfBatches: number;
  unbatchedFeatures: Matrix;
  unbatchedLabels: Matrix;
}

export interface TrainingProps {
  trainingFeatureBatches: Matrix[];
  trainingLabelBatches: Matrix[];
  numberOfHiddenNodes: number;
  learningRate: number;
  epochs: number;
  showLoss: boolean | undefined;
  testingFeatures: Matrix;
  testingLabels: Matrix;
  unbatchedTrainingFeatures: Matrix;
  unbatchedTrainingLabels: Matrix;
}
