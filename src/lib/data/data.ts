import fs from "fs";
import { Matrix } from "../functions/matrix";
import { splitArrayInHalf } from "./utility";
import { TrainingData } from "./interfaces";
import { BaseData } from "./base-data";

export class Data extends BaseData {
  public static loadTraining(
    filePath: string,
    batchSize = 0,
    oneHotEncode = false,
  ): TrainingData {
    try {
      const fileContent = fs.readFileSync(filePath, "utf8");
      const rows = fileContent.split("\n");
      const features: number[][] = [];
      const labels: number[][] = [];

      rows.map((row) => {
        const values = row.split(",").map((value) => {
          const number = parseFloat(value);
          return number;
        });

        features.push(values.slice(0, -1));
        labels.push([values[values.length - 1]]);
      });

      const featuresMatrix = new Matrix(features);

      const labelsMatrix = oneHotEncode
        ? this.oneHotEncode(labels)
        : new Matrix(labels);

      if (batchSize > 0) {
        const featuresBatches: number[][][] = [];
        const labelsBatches: number[][][] = [];
        const numberOfBatches = Math.floor(features.length / batchSize);
        const lastBatchSize = features.length % batchSize;

        for (let i = 0; i < numberOfBatches; i++) {
          featuresBatches.push(
            features.slice(i * batchSize, (i + 1) * batchSize),
          );
          labelsBatches.push(labels.slice(i * batchSize, (i + 1) * batchSize));
        }

        if (lastBatchSize > 0) {
          featuresBatches.push(features.slice(-lastBatchSize));
          labelsBatches.push(labels.slice(-lastBatchSize));
        }

        return {
          trainingFeatureBatches: featuresBatches.map(
            (batch) => new Matrix(batch),
          ),
          trainingLabelBatches: labelsBatches.map((batch) => new Matrix(batch)),
          batchSize,
          lastBatchSize,
          numberOfBatches,
          unbatchedTrainingFeatures: featuresMatrix,
          unbatchedTrainingLabels: labelsMatrix,
        };
      }

      return {
        trainingFeatureBatches: [featuresMatrix],
        trainingLabelBatches: [labelsMatrix],
        batchSize: 0,
        lastBatchSize: featuresMatrix.rows,
        numberOfBatches: 1,
        unbatchedTrainingFeatures: featuresMatrix,
        unbatchedTrainingLabels: labelsMatrix,
      };
    } catch (error) {
      throw new Error("Error reading the CSV file");
    }
  }

  public static loadValidationAndTest(filePath: string) {
    try {
      const fileContent = fs.readFileSync(filePath, "utf8");
      const rows = fileContent.split("\n");
      const features: number[][] = [];
      const labels: number[][] = [];

      rows.map((row) => {
        const values = row.split(",").map((value) => {
          const number = parseFloat(value);
          return number;
        });

        features.push(values.slice(0, -1));
        labels.push([values[values.length - 1]]);
      });

      const { firstHalf: testFeatures, secondHalf: validateFeatures } =
        splitArrayInHalf(features);

      const { firstHalf: testLabels, secondHalf: validateLabels } =
        splitArrayInHalf(labels);

      return {
        testFeatures: new Matrix(testFeatures),
        validationFeatures: new Matrix(validateFeatures),
        testLabels: new Matrix(testLabels),
        validationLabels: new Matrix(validateLabels),
      };
    } catch (error) {
      throw new Error("Error reading the CSV file");
    }
  }
}
