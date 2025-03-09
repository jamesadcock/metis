import { Matrix } from "../functions/matrix";

export abstract class BaseData {
  protected static oneHotEncode(labels: number[][]): Matrix {
    const max = Math.max(...labels.flat());
    const encodedLabels = labels.map((label) => {
      const oneHotEncoded = Array(max + 1).fill(0);
      oneHotEncoded[label[0]] = 1;
      return oneHotEncoded;
    });

    return new Matrix(encodedLabels);
  }
}
