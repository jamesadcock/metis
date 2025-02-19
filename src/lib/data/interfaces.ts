import { Matrix } from "../functions/matrix";

export interface TrainingData {
  features: Matrix[];
  labels: Matrix[];
  batchSize: number;
  lastBatchSize: number;
  numberOfBatches: number;
  unbatchedFeatures: Matrix;
  unbatchedLabels: Matrix;
}
