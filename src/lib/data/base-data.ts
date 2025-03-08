import { Matrix } from "../functions/matrix";

export abstract class BaseData {
  protected static oneHotEncode(labels: number[][]): Matrix {
    const uniqueLabels = Array.from(new Set(labels.flat()));
    const oneHotEncodedLabels = labels.map((label) => {
      const oneHotEncodedLabel = new Array(uniqueLabels.length).fill(0);
      const index = uniqueLabels.indexOf(label[0]);
      oneHotEncodedLabel[index] = 1;
      return oneHotEncodedLabel;
    });

    return new Matrix(oneHotEncodedLabels);
  }
}
