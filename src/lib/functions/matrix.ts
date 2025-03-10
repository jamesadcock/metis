export class Matrix {
  private data: number[][];

  constructor(data: number[][]) {
    this.data = data;
  }

  public static fromTypedArray(typedData: Float64Array[]): Matrix {
    const convertedData: number[][] = Array.from(typedData, (row) =>
      Array.from(row),
    );

    return new Matrix(convertedData);
  }

  get() {
    return this.data;
  }

  get rows() {
    return this.data.length;
  }

  get columns() {
    return this.data[0].length;
  }

  public addColumn(column: number[], position: number): Matrix {
    const result: number[][] = [];
    for (let i = 0; i < this.data.length; i++) {
      result[i] = Array.from(this.data[i]);
      result[i].splice(position, 0, column[i]);
    }
    return new Matrix(result);
  }

  public multiplyMatrices(matrix: Matrix): Matrix {
    const rowsA = this.data.length;
    const colsA = this.data[0].length;
    const colsB = matrix.data[0].length;

    if (colsA !== matrix.data.length) {
      throw new Error(
        `Invalid matrix size: ${this.data.length}x${this.data[0].length} and ${matrix.data.length}x${matrix.data[0].length}`,
      );
    }

    const result = Array.from({ length: rowsA }, () =>
      new Float64Array(colsB).fill(0),
    );
    const matrixData = matrix.data;

    for (let i = 0; i < rowsA; i++) {
      for (let k = 0; k < colsA; k++) {
        const temp = this.data[i][k];
        for (let j = 0; j < colsB; j++) {
          result[i][j] += temp * matrixData[k][j];
        }
      }
    }

    return Matrix.fromTypedArray(result);
  }

  public multiply(multiplier: number): Matrix {
    const result: number[][] = [];
    for (let i = 0; i < this.data.length; i++) {
      result[i] = [];
      for (let j = 0; j < this.data[0].length; j++) {
        result[i][j] = this.data[i][j] * multiplier;
      }
    }
    return new Matrix(result);
  }

  public elementWiseMultiplication(matrix: Matrix): Matrix {
    if (
      this.data.length !== matrix.data.length ||
      this.data[0].length !== matrix.data[0].length
    ) {
      throw new Error(
        `Invalid matrix size: ${this.data.length}x${this.data[0].length} and ${matrix.data.length}x${matrix.data[0].length}`,
      );
    }

    const result: number[][] = [];
    for (let i = 0; i < this.data.length; i++) {
      result[i] = [];
      for (let j = 0; j < this.data[0].length; j++) {
        result[i][j] = this.data[i][j] * matrix.data[i][j];
      }
    }
    return new Matrix(result);
  }

  public transpose(): Matrix {
    const result: number[][] = [];
    for (let i = 0; i < this.data[0].length; i++) {
      result[i] = [];
      for (let j = 0; j < this.data.length; j++) {
        result[i][j] = this.data[j][i];
      }
    }
    return new Matrix(result);
  }

  public subtractMatrices(matrix: Matrix): Matrix {
    if (
      this.data.length !== matrix.data.length ||
      this.data[0].length !== matrix.data[0].length
    ) {
      throw new Error(
        `Invalid matrix size: ${this.data.length}x${this.data[0].length} and ${matrix.data.length}x${matrix.data[0].length}`,
      );
    }

    const result: number[][] = [];
    for (let i = 0; i < this.data.length; i++) {
      result[i] = [];
      for (let j = 0; j < this.data[0].length; j++) {
        result[i][j] = this.data[i][j] - matrix.data[i][j];
      }
    }
    return new Matrix(result);
  }

  public subtract(number: number): Matrix {
    const result: number[][] = [];
    for (let i = 0; i < this.data.length; i++) {
      result[i] = [];
      for (let j = 0; j < this.data[0].length; j++) {
        result[i][j] = this.data[i][j] - number;
      }
    }
    return new Matrix(result);
  }

  public addMatrices(matrix: Matrix): Matrix {
    if (
      this.data.length !== matrix.data.length ||
      this.data[0].length !== matrix.data[0].length
    ) {
      throw new Error("Invalid matrix size");
    }

    const result: number[][] = [];
    for (let i = 0; i < this.data.length; i++) {
      result[i] = [];
      for (let j = 0; j < this.data[0].length; j++) {
        result[i][j] = this.data[i][j] + matrix.data[i][j];
      }
    }
    return new Matrix(result);
  }

  public squared(): Matrix {
    const result: number[][] = [];
    for (let i = 0; i < this.data.length; i++) {
      result[i] = [];
      for (let j = 0; j < this.data[0].length; j++) {
        result[i][j] = Math.pow(this.data[i][j], 2);
      }
    }
    return new Matrix(result);
  }

  public mean(): number {
    let sum = 0;
    for (let i = 0; i < this.data.length; i++) {
      for (let j = 0; j < this.data[0].length; j++) {
        sum += this.data[i][j];
      }
    }
    return sum / (this.data.length * this.data[0].length);
  }

  public standardDeviation(): number {
    const mean = this.mean();
    const sum = this.data
      .map((row) => row.map((value) => Math.pow(value - mean, 2)))
      .reduce((acc, row) => acc.concat(row), [])
      .reduce((acc, value) => acc + value, 0);
    return Math.sqrt(sum / (this.data.length * this.data[0].length));
  }

  public divide(divisor: number): Matrix {
    const result: number[][] = [];
    for (let i = 0; i < this.data.length; i++) {
      result[i] = [];
      for (let j = 0; j < this.data[0].length; j++) {
        result[i][j] = this.data[i][j] / divisor;
      }
    }
    return new Matrix(result);
  }

  public applyFunction(fn: (input: number) => number): Matrix {
    const result: number[][] = [];
    for (let i = 0; i < this.data.length; i++) {
      result[i] = [];
      for (let j = 0; j < this.data[0].length; j++) {
        result[i][j] = fn(this.data[i][j]);
      }
    }

    return new Matrix(result);
  }

  public max(): Matrix {
    const result: number[][] = [];
    for (let i = 0; i < this.data.length; i++) {
      result[i] = [Math.max(...this.data[i])];
    }
    return new Matrix(result);
  }

  public matrixMax(): number {
    let max = -Infinity;
    for (let i = 0; i < this.data.length; i++) {
      for (let j = 0; j < this.data[0].length; j++) {
        max = Math.max(max, this.data[i][j]);
      }
    }
    return max;
  }

  /*
  The argMax function finds the index of the maximum value in each
  row of the matrix and returns a new matrix containing these indices.
  */
  public argMax(): Matrix {
    const result: number[][] = [];
    for (let i = 0; i < this.data.length; i++) {
      result[i] = [this.data[i].indexOf(Math.max(...this.data[i]))];
    }
    return new Matrix(result);
  }

  public sum(): number {
    let sum = 0;
    for (let i = 0; i < this.data.length; i++) {
      for (let j = 0; j < this.data[0].length; j++) {
        sum += this.data[i][j];
      }
    }
    return sum;
  }

  public sumRows(): Matrix {
    const result: number[][] = [];
    for (let i = 0; i < this.data.length; i++) {
      result[i] = [this.data[i].reduce((acc, val) => acc + val, 0)];
    }
    return new Matrix(result);
  }

  public divideRow(matrix: Matrix): Matrix {
    if (matrix.columns !== 1 || matrix.rows !== this.rows) {
      throw new Error("Invalid matrix size");
    }

    const result: number[][] = [];
    for (let i = 0; i < this.data.length; i++) {
      result[i] = [];
      for (let j = 0; j < this.data[0].length; j++) {
        result[i][j] = this.data[i][j] / matrix.data[i][0];
      }
    }
    return new Matrix(result);
  }

  public removeRow(row: number): Matrix {
    const result = this.data.slice();
    result.splice(row, 1);
    return new Matrix(result);
  }

  public removeColumn(column: number): Matrix {
    const result = this.data.slice();
    for (let i = 0; i < result.length; i++) {
      result[i].splice(column, 1);
    }
    return new Matrix(result);
  }

  public static random(rows: number, columns: number): Matrix {
    const result: number[][] = [];
    for (let i = 0; i < rows; i++) {
      result[i] = [];
      for (let j = 0; j < columns; j++) {
        result[i][j] = Math.random() * 2 - 1;
      }
    }
    return new Matrix(result);
  }
}
