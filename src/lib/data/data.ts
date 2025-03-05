import fs from "fs";
import { Matrix } from "../functions/matrix";
import { splitArrayInHalf } from "./utility";
import { IData } from "./interfaces";

export class Data {
  public static loadTraining(filePath: string, batchSize = 0): IData {
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
      const labelsMatrix = new Matrix(labels);
      if (batchSize > 0) {
        const featuresBatches: number[][][] = [];
        const labelsBatches: number[][][] = [];
        const numberOfBatches = Math.floor(features.length / batchSize);
        const lastBatchSize = features.length % batchSize;

        for (let i = 0; i < numberOfBatches; i++) {
          featuresBatches.push(
            features.slice(i * batchSize, (i + 1) * batchSize)
          );
          labelsBatches.push(labels.slice(i * batchSize, (i + 1) * batchSize));
        }

        if (lastBatchSize > 0) {
          featuresBatches.push(features.slice(-lastBatchSize));
          labelsBatches.push(labels.slice(-lastBatchSize));
        }

        return {
          trainingFeatures: featuresBatches.map((batch) => new Matrix(batch)),
          trainingLabels: labelsBatches.map((batch) => new Matrix(batch)),
          batchSize,
          lastBatchSize,
          numberOfBatches,
          unbatchedFeatures: featuresMatrix,
          unbatchedLabels: labelsMatrix,
        };
      }

      return {
        trainingFeatures: [featuresMatrix],
        trainingLabels: [labelsMatrix],
        batchSize: 0,
        lastBatchSize: featuresMatrix.rows,
        numberOfBatches: 1,
        unbatchedFeatures: featuresMatrix,
        unbatchedLabels: labelsMatrix,
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
