import * as fs from "fs";
import { Matrix } from "../functions/matrix";
import { TrainingData } from "./interfaces";
import { splitArrayInHalf } from "./utility";

export class Mnist {
  public load(
    trainingImagePath: string,
    trainingLabelsPath: string,
    testingImagePath: string,
    testingLabelsPath: string,
    batchSize = 0
  ): TrainingData {
    const { testingLabels, validationLabels } =
      this.loadTestAndValidationLabels(testingLabelsPath);
    const { testingFeatures: testingFeaturesRaw, validationFeatures } =
      this.loadTestAndValidationImages(testingImagePath);

    const featuresRaw = this.loadImages(trainingImagePath);
    const {
      trainingSetStandardized: features,
      testSetStandardized: testingFeatures,
    } = this.standardize(featuresRaw, testingFeaturesRaw);
    const labels = this.loadTrainingLabels(trainingLabelsPath);

    if (batchSize === 0) {
      const featuresMatrix = new Matrix(features);
      const trainingLabels = this.oneHotEncode(labels);
      return {
        trainingFeatures: [featuresMatrix],
        trainingLabels: [trainingLabels],
        batchSize: 0,
        lastBatchSize: featuresMatrix.rows,
        numberOfBatches: 1,
        unbatchedFeatures: featuresMatrix,
        unbatchedLabels: trainingLabels,
        testingFeatures: new Matrix(testingFeatures),
        testingLabels,
        validationFeatures: new Matrix(validationFeatures),
        validationLabels,
      };
    }

    const featuresBatches: number[][][] = [];
    const labelsBatches: number[][][] = [];

    const numberOfBatches = Math.floor(features.length / batchSize);
    let lastBatchSize = features.length % batchSize;

    for (let i = 0; i < numberOfBatches; i++) {
      featuresBatches.push(features.slice(i * batchSize, (i + 1) * batchSize));
      labelsBatches.push(labels.slice(i * batchSize, (i + 1) * batchSize));
    }

    if (lastBatchSize > 0) {
      featuresBatches.push(features.slice(-lastBatchSize));
      labelsBatches.push(labels.slice(-lastBatchSize));
    } else {
      lastBatchSize = batchSize;
    }

    return {
      trainingFeatures: featuresBatches.map((batch) => new Matrix(batch)),
      trainingLabels: labelsBatches.map((batch) => this.oneHotEncode(batch)),
      batchSize,
      lastBatchSize,
      numberOfBatches,
      unbatchedFeatures: new Matrix(features),
      unbatchedLabels: this.oneHotEncode(labels),
      testingFeatures: new Matrix(testingFeatures),
      testingLabels,
      validationFeatures: new Matrix(validationFeatures),
      validationLabels,
    };
  }

  // Load images from the specified file and return as a Matrix
  private loadImages(filename: string) {
    const buffer = this.readBuffer(filename);
    const headerSize = 16; // Header size for image files
    const imageSize = 28 * 28; // Each image is 28x28 pixels
    const images: number[][] = [];

    // Read each image from the buffer
    for (let i = headerSize; i < buffer.length; i += imageSize) {
      const image: number[] = [];
      for (let j = 0; j < imageSize; j++) {
        image.push(buffer[i + j]);
      }
      images.push(image);
    }

    return images;
  }

  // Load training labels from the specified file and return as a one-hot encoded Matrix
  private loadTrainingLabels(filename: string) {
    const buffer = this.readBuffer(filename);
    const headerSize = 8; // Header size for label files
    const labels: number[][] = [];

    // Read each label from the buffer
    for (let i = headerSize; i < buffer.length; i++) {
      labels.push([buffer[i]]);
    }

    return labels;
  }

  private loadTestAndValidationLabels(filename: string) {
    const buffer = this.readBuffer(filename);
    const headerSize = 8;
    const labels: number[][] = [];

    for (let i = headerSize; i < buffer.length; i++) {
      labels.push([buffer[i]]);
    }

    const { firstHalf: validation, secondHalf: test } =
      splitArrayInHalf(labels);
    return {
      testingLabels: new Matrix(test),
      validationLabels: new Matrix(validation),
    };
  }

  private loadTestAndValidationImages(filename: string) {
    const images = this.loadImages(filename);
    const { firstHalf: validation, secondHalf: test } =
      splitArrayInHalf(images);
    return {
      testingFeatures: test,
      validationFeatures: validation,
    };
  }
 
  // One-hot encode the labels and return as a Matrix
  protected oneHotEncode(labels: number[][]): Matrix {
    const encodedLabels = labels.map((label) => {
      const oneHotEncoded = Array(10).fill(0);
      oneHotEncoded[label[0]] = 1;
      return oneHotEncoded;
    });

    return new Matrix(encodedLabels);
  }

  // Read the file and return its content as a Buffer
  private readBuffer(filename: string): Buffer {
    try {
      return fs.readFileSync(filename);
    } catch (e) {
      throw new Error(`File not found: ${filename}`);
    }
  }

  protected standardize(trainingSet: number[][], testSet: number[][]) {
    const flattenedTrainingSet = trainingSet.flat();
    const average =
      flattenedTrainingSet.reduce((a, b) => a + b, 0) /
      flattenedTrainingSet.length;

    let sumOfSquaredDifferences = 0;
    for (let i = 0; i < flattenedTrainingSet.length; i++) {
      sumOfSquaredDifferences += Math.pow(flattenedTrainingSet[i] - average, 2);
    }
    const standardDeviation = Math.sqrt(
      sumOfSquaredDifferences / flattenedTrainingSet.length
    );

    const standardizeRow = (row: number[]): number[] => {
      const standardizedRow: number[] = new Array(row.length);
      for (let i = 0; i < row.length; i++) {
        standardizedRow[i] = (row[i] - average) / standardDeviation;
      }
      return standardizedRow;
    };

    const trainingSetStandardized = trainingSet.map(standardizeRow);
    const testSetStandardized = testSet.map(standardizeRow);
    return { trainingSetStandardized, testSetStandardized };
  }
}
