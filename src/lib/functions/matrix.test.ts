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

describe("rows", () => {
  it("should return correct number of rows", () => {
    const matrix = new Matrix([
      [1, 2],
      [3, 4],
      [5, 6],
    ]);
    expect(matrix.rows).toEqual(3);
  });
});

describe("columns", () => {
  it("should return correct number of columns", () => {
    const matrix = new Matrix([
      [1, 2],
      [3, 4],
      [5, 6],
    ]);
    expect(matrix.columns).toEqual(2);
  });
});

describe("multiplyMatrices", () => {
  it("should multiply (1 x 3) and (3 x 1) return correct result", () => {
    const matrixA = new Matrix([[2, 3, 5]]);

    const matrixB = new Matrix([[2.5], [4], [1]]);

    const result = matrixA.multiplyMatrices(matrixB);
    expect(result.get()).toEqual([[22]]);
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

    const result = matrixA.multiplyMatrices(matrixB);
    expect(result.get()).toEqual([
      [22, 40],
      [98.5, 161],
      [186.5, 233],
      [57.5, 195],
    ]);
  });

  it("should throw error when multiply (1 x 3) and (2 x 1)", () => {
    const matrixA = new Matrix([[2, 3, 5]]);

    const matrixB = new Matrix([[2], [4]]);

    expect(() => matrixA.multiplyMatrices(matrixB)).toThrowError(
      "Invalid matrix size"
    );
  });
});

describe("multiply", () => {
  it("should multiply (2 x 2) matrix by 2", () => {
    const matrix = new Matrix([
      [1, 2],
      [3, 4],
    ]);

    const result = matrix.multiply(2);
    expect(result.get()).toEqual([
      [2, 4],
      [6, 8],
    ]);
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
    expect(result.get()).toEqual([
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
    expect(result.get()).toEqual([
      [1, 2],
      [3, 4],
      [5, 6],
    ]);
  });
});

describe("subtract", () => {
  it("should subtract (5 x 1) matrix", () => {
    const matrixA = new Matrix([[1], [3], [5], [7], [9]]);

    const matrixB = new Matrix([[2], [4], [6], [8], [10]]);

    const result = matrixA.subtract(matrixB);
    expect(result.get()).toEqual([[-1], [-1], [-1], [-1], [-1]]);
  });

  it("should subtract (2 x 2) matrix", () => {
    const matrixA = new Matrix([
      [1, 2],
      [3, 4],
    ]);

    const matrixB = new Matrix([
      [2, 1],
      [4, 3],
    ]);

    const result = matrixA.subtract(matrixB);
    expect(result.get()).toEqual([
      [-1, 1],
      [-1, 1],
    ]);
  });

  it("should throw error when subtract (2 x 2) and (2 x 3) matrix", () => {
    const matrixA = new Matrix([
      [1, 2],
      [3, 4],
    ]);

    const matrixB = new Matrix([
      [2, 1, 3],
      [4, 3, 5],
    ]);

    expect(() => matrixA.subtract(matrixB)).toThrowError("Invalid matrix size");
  });
});

describe("squared", () => {
  it("should return squared matrix", () => {
    const matrix = new Matrix([
      [1, 2],
      [3, 4],
    ]);

    const result = matrix.squared();
    expect(result.get()).toEqual([
      [1, 4],
      [9, 16],
    ]);
  });
});

describe("mean", () => {
  it("should return mean of matrix", () => {
    const matrix = new Matrix([[1], [2], [3]]);

    const result = matrix.mean();
    expect(result).toEqual(2);
  });
});

describe("divide", () => {
  it("should divide matrix by number", () => {
    const matrix = new Matrix([
      [2, 4],
      [6, 8],
    ]);

    const result = matrix.divide(2);
    expect(result.get()).toEqual([
      [1, 2],
      [3, 4],
    ]);
  });
});
