import * as fs from "fs";
import { Matrix } from "../functions/matrix";

export class Mnist {
  public loadImages(filename: string): Matrix {
    const buffer = this.readBuffer(filename);
    const headerSize = 16;
    const imageSize = 28 * 28;
    const images = [];

    for (let i = headerSize; i < buffer.length; i += imageSize) {
      const image = [];
      for (let j = 0; j < imageSize; j++) {
        image.push(buffer[i + j]);
      }
      images.push(image);
    }

    return new Matrix(images);
  }

  public loadTrainingLabels(filename: string) {
    const buffer = this.readBuffer(filename);
    const headerSize = 8;
    const labels = [];

    for (let i = headerSize; i < buffer.length; i++) {
      labels.push(buffer[i]);
    }

    return this.oneHotEncode(labels);
  }

  public loadTestLabels(filename: string) {
    const buffer = this.readBuffer(filename);
    const headerSize = 8;
    const labels = new Array<Array<number>>();

    for (let i = headerSize; i < buffer.length; i++) {
      labels.push([buffer[i]]);
    }

    return new Matrix(labels);
  }

  public oneHotEncode(labels: number[]): Matrix {
    const encodedLabels = labels.map((label) => {
      const oneHotEncoded = Array(10).fill(0);
      oneHotEncoded[label] = 1;
      return oneHotEncoded;
    });

    return new Matrix(encodedLabels);
  }

  private readBuffer(filename: string): Buffer {
    let buffer: string | unknown[] | Buffer;
    try {
      buffer = fs.readFileSync(filename);
    } catch (e) {
      throw new Error(`File not found: ${filename}`);
    }
    return buffer as Buffer;
  }
}
