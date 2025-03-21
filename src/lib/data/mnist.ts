import * as fs from "fs";
import { Matrix } from "../functions/matrix";
import { TrainingData } from "./interfaces";
import { splitArrayInHalf } from "./utility";
import { BaseData } from "./base-data";
import { createCanvas } from "canvas";

export class Mnist extends BaseData {
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
      const trainingLabels = Mnist.oneHotEncode(labels);
      return {
        trainingFeatureBatches: [featuresMatrix],
        trainingLabelBatches: [trainingLabels],
        batchSize: 0,
        lastBatchSize: featuresMatrix.rows,
        numberOfBatches: 1,
        unbatchedTrainingFeatures: featuresMatrix,
        unbatchedTrainingLabels: trainingLabels,
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
      trainingFeatureBatches: featuresBatches.map((batch) => new Matrix(batch)),
      trainingLabelBatches: labelsBatches.map((batch) =>
        Mnist.oneHotEncode(batch)
      ),
      batchSize,
      lastBatchSize,
      numberOfBatches,
      unbatchedTrainingFeatures: new Matrix(features),
      unbatchedTrainingLabels: Mnist.oneHotEncode(labels),
      testingFeatures: new Matrix(testingFeatures),
      testingLabels,
      validationFeatures: new Matrix(validationFeatures),
      validationLabels,
    };
  }

  public loadTestImages() {
    const { testingFeatures } = this.loadTestAndValidationImages(
      "test-data/mnist/t10k-images-idx3-ubyte"
    );
    const featuresRaw = this.loadImages(
      "test-data/mnist/train-images-idx3-ubyte"
    );

    const standardizedImages = this.standardize(
      featuresRaw,
      testingFeatures
    ).testSetStandardized;
    return new Matrix(standardizedImages);
  }

  // Load images from the specified file and return as a Matrix
  public loadImages(filename: string) {
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

    const standardizeNumber = (number: number[]): number[] => {
      const standardizedNumber: number[] = new Array(number.length);
      for (let i = 0; i < number.length; i++) {
        standardizedNumber[i] = (number[i] - average) / standardDeviation;
      }
      return standardizedNumber;
    };

    const trainingSetStandardized = trainingSet.map(standardizeNumber);
    const testSetStandardized = testSet.map(standardizeNumber);
    return { trainingSetStandardized, testSetStandardized };
  }

  public renderImage(image: number[]) {
    // Create a canvas for the image
    const canvas = createCanvas(28, 28);
    const ctx = canvas.getContext("2d");

    // Draw the first image onto the canvas
    for (let y = 0; y < 28; y++) {
      for (let x = 0; x < 28; x++) {
        const pixel = image[y * 28 + x] * 255; // Scale to 0-255
        ctx.fillStyle = `rgb(${pixel}, ${pixel}, ${pixel})`;
        ctx.fillRect(x, y, 1, 1);
      }
    }

    // Save the image to a file
    const buffer = canvas.toBuffer("image/png");
    fs.writeFileSync("digit.png", buffer);
    console.log("Image saved to digit.png");
  }
}