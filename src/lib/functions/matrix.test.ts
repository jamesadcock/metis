import { Matrix } from "./matrix";

describe("get", () => {
  it("should return correct matrix data", () => {
    const matrix = new Matrix([
      [1, 2],
      [3, 4],
      [5, 6],
    ]);
    expect(matrix.get()).toEqual([
      [1, 2],
      [3, 4],
      [5, 6],
    ]);
  });
});

describe("multiply", () => {
  it("should multiply (1 x 3) and (3 x 1) return correct result", () => {
    const matrixA = new Matrix([[2, 3, 5]]);

    const matrixB = new Matrix([[2.5], [4], [1]]);

    const result = matrixA.multiply(matrixB);
    expect(result).toEqual([[22]]);
  });

  it("should multiply (4 x 3) and (3 x 2) return correct result", () => {
    const matrixA = new Matrix([
      [2, 3, 5],
      [11, 13, 19],
      [31, 27, 1],
      [-3, 14, 9],
    ]);

    const matrixB = new Matrix([
      [2.5, -3],
      [4, 12],
      [1, 2],
    ]);

    const result = matrixA.multiply(matrixB);
    expect(result).toEqual([
      [22, 40],
      [98.5, 161],
      [186.5, 233],
      [57.5, 195],
    ]);
  });

  it("should throw error when multiply (1 x 3) and (2 x 1)", () => {
    const matrixA = new Matrix([[2, 3, 5]]);

    const matrixB = new Matrix([[2], [4]]);

    expect(() => matrixA.multiply(matrixB)).toThrowError("Invalid matrix size");
  });
});

describe("transpose", () => {
  it("should transpose (3 x 2) matrix", () => {
    const matrix = new Matrix([
      [1, 2],
      [3, 4],
      [5, 6],
    ]);

    const result = matrix.transpose();
    expect(result).toEqual([
      [1, 3, 5],
      [2, 4, 6],
    ]);
  });

  it("should transpose (2 x 3) matrix", () => {
    const matrix = new Matrix([
      [1, 3, 5],
      [2, 4, 6],
    ]);

    const result = matrix.transpose();
    expect(result).toEqual([
      [1, 2],
      [3, 4],
      [5, 6],
    ]);
  });
});
