import { describe } from "node:test";
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
      "Invalid matrix size",
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

describe("add", () => {
  it("should add (5 x 1) matrix", () => {
    const matrixA = new Matrix([[1], [3], [5], [7], [9]]);
    const matrixB = new Matrix([[2], [4], [6], [8], [10]]);
    const result = matrixA.addMatrices(matrixB);
    expect(result.get()).toEqual([[3], [7], [11], [15], [19]]);
  });

  it("should add (2 x 2) matrix", () => {
    const matrixA = new Matrix([
      [1, 2],
      [3, 4],
    ]);
    const matrixB = new Matrix([
      [2, 1],
      [4, 3],
    ]);
    const result = matrixA.addMatrices(matrixB);
    expect(result.get()).toEqual([
      [3, 3],
      [7, 7],
    ]);
  });

  it("should throw error when add (2 x 2) and (2 x 3) matrix", () => {
    const matrixA = new Matrix([
      [1, 2],
      [3, 4],
    ]);
    const matrixB = new Matrix([
      [2, 1, 3],
      [4, 3, 5],
    ]);
    expect(() => matrixA.addMatrices(matrixB)).toThrowError(
      "Invalid matrix size",
    );
  });
});

describe("subtractMatrices", () => {
  it("should subtract (5 x 1) matrix", () => {
    const matrixA = new Matrix([[1], [3], [5], [7], [9]]);

    const matrixB = new Matrix([[2], [4], [6], [8], [10]]);

    const result = matrixA.subtractMatrices(matrixB);
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

    const result = matrixA.subtractMatrices(matrixB);
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

    expect(() => matrixA.subtractMatrices(matrixB)).toThrowError(
      "Invalid matrix size",
    );
  });
});

describe("subtract", () => {
  it("should subtract (2 x 2) matrix by 2", () => {
    const matrix = new Matrix([
      [1, 2],
      [3, 4],
    ]);

    const result = matrix.subtract(2);
    expect(result.get()).toEqual([
      [-1, 0],
      [1, 2],
    ]);
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

describe("applyFunction", () => {
  it("should apply function to matrix", () => {
    const matrix = new Matrix([
      [1, 2],
      [3, 4],
    ]);

    const result = matrix.applyFunction((input) => input * 2);
    expect(result.get()).toEqual([
      [2, 4],
      [6, 8],
    ]);
  });

  it("apply 1 minus input function", () => {
    const matrix = new Matrix([
      [1, 2],
      [3, 4],
    ]);

    const result = matrix.applyFunction((input) => 1 - input);
    expect(result.get()).toEqual([
      [0, -1],
      [-2, -3],
    ]);
  });
});

describe("elementWiseMultiplication", () => {
  it("should multiply two matrices element-wise", () => {
    const matrixA = new Matrix([
      [1, 2],
      [3, 4],
    ]);

    const matrixB = new Matrix([
      [2, 3],
      [4, 5],
    ]);

    const result = matrixA.elementWiseMultiplication(matrixB);
    expect(result.get()).toEqual([
      [2, 6],
      [12, 20],
    ]);
  });

  it("should throw error when multiplying two matrices of different sizes", () => {
    const matrixA = new Matrix([
      [1, 2],
      [3, 4],
    ]);

    const matrixB = new Matrix([
      [2, 3, 4],
      [4, 5, 6],
    ]);

    expect(() => matrixA.elementWiseMultiplication(matrixB)).toThrowError(
      "Invalid matrix size",
    );
  });
});

describe("addColumn", () => {
  it("should add a column to a matrix", () => {
    const matrix = new Matrix([
      [1, 2],
      [3, 4],
    ]);

    const column = [5, 6];

    const result = matrix.addColumn(column);
    expect(result.get()).toEqual([
      [1, 2, 5],
      [3, 4, 6],
    ]);
  });
});
