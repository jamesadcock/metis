import fs from "fs";
import { Matrix } from "../functions/matrix";
import { TrainingData } from "./interfaces";

export class Data {
  public static loadTraining(filePath: string, batchSize = 0): TrainingData {
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
          features: featuresBatches.map((batch) => new Matrix(batch)),
          labels: labelsBatches.map((batch) => new Matrix(batch)),
          batchSize,
          lastBatchSize,
          numberOfBatches,
        };
      }

      const featuresMatrix = new Matrix(features);
      const labelsMatrix = new Matrix(labels);

      return {
        features: [featuresMatrix],
        labels: [labelsMatrix],
        batchSize: 0,
        lastBatchSize: featuresMatrix.rows,
        numberOfBatches: 1,
      };
    } catch (error) {
      throw new Error("Error reading the CSV file");
    }
  }

  public static load(filePath: string) {
    try {
      const fileContent = fs.readFileSync(filePath, "utf8");
      const rows = fileContent.split("\n");
      const data: number[][] = [];

      rows.map((row) => {
        const values = row.split(",").map((value) => {
          const number = parseFloat(value);
          return number;
        });

        data.push(values);
      });

      return { data: new Matrix(data) };
    } catch (error) {
      throw new Error("Error reading the CSV file");
    }
  }
}
