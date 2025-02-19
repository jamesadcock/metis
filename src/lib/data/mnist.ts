import * as fs from "fs";
import { Matrix } from "../functions/matrix";
import { TrainingData } from "./interfaces";
import { splitArrayInHalf } from "./utility";

export class Mnist {
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

  public loadTrainingData(
    imagePath: string,
    labelsPath: string,
    batchSize = 0,
  ): TrainingData {
    if (batchSize === 0) {
      const features = new Matrix(this.loadImages(imagePath));
      const labels = this.oneHotEncode(this.loadTrainingLabels(labelsPath));
      return {
        features: [features],
        labels: [labels],
        batchSize: 0,
        lastBatchSize: features.rows,
        numberOfBatches: 1,
        unbatchedFeatures: features,
        unbatchedLabels: labels,
      };
    }

    const features = this.loadImages(imagePath);
    const labels = this.loadTrainingLabels(labelsPath);
    const featuresBatches: number[][][] = [];
    const labelsBatches: number[][][] = [];

    const numberOfBatches = Math.floor(features.length / batchSize);
    const lastBatchSize = features.length % batchSize;

    for (let i = 0; i < numberOfBatches; i++) {
      featuresBatches.push(features.slice(i * batchSize, (i + 1) * batchSize));
      labelsBatches.push(labels.slice(i * batchSize, (i + 1) * batchSize));
    }

    if (lastBatchSize > 0) {
      featuresBatches.push(features.slice(-lastBatchSize));
      labelsBatches.push(labels.slice(-lastBatchSize));
    }

    return {
      features: featuresBatches.map((batch) => new Matrix(batch)),
      labels: labelsBatches.map((batch) => this.oneHotEncode(batch)),
      batchSize,
      lastBatchSize,
      numberOfBatches,
      unbatchedFeatures: new Matrix(features),
      unbatchedLabels: this.oneHotEncode(labels),
    };
  }

  public loadTestAndValidationLabels(filename: string) {
    const buffer = this.readBuffer(filename);
    const headerSize = 8;
    const labels: number[][] = [];

    for (let i = headerSize; i < buffer.length; i++) {
      labels.push([buffer[i]]);
    }

    const { firstHalf: test, secondHalf: validate } = splitArrayInHalf(labels);
    return {
      test: new Matrix(test),
      validate: new Matrix(validate),
    };
  }

  public loadTestAndValidationImages(filename: string) {
    const images = this.loadImages(filename);
    const { firstHalf: test, secondHalf: validate } = splitArrayInHalf(images);
    return { test: new Matrix(test), validate: new Matrix(validate) };
  }

  // One-hot encode the labels and return as a Matrix
  public oneHotEncode(labels: number[][]): Matrix {
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
}
