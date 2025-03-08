import { Matrix } from "../functions/matrix";
import { BaseData } from "./base-data";

describe("oneHotEncode", () => {
  it("should return a matrix with one-hot encoded labels", () => {
    const labels = [[0], [1], [2], [0], [1], [2]];
    const expected = new Matrix([
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ]);

    const result = TestData.oneHotEncode(labels);
    expect(result).toEqual(expected);
  });
});

class TestData extends BaseData {
  public static oneHotEncode(labels: number[][]): Matrix {
    return super.oneHotEncode(labels);
  }
}
