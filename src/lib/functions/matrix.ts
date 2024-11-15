export class Matrix {
  private data: number[][];

  constructor(data: number[][]) {
    this.data = data;
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

  public addColumn(column: number[]): Matrix {
    if (column.length !== this.data.length) {
      throw new Error("Invalid column size");
    }

    const result: number[][] = [];
    for (let i = 0; i < this.data.length; i++) {
      result[i] = this.data[i].concat(column[i]);
    }
    return new Matrix(result);
  }

  public multiplyMatrices(matrix: Matrix): Matrix {
    if (this.data[0].length !== matrix.data.length) {
      throw new Error("Invalid matrix size");
    }

    const result: number[][] = [];
    for (let i = 0; i < this.data.length; i++) {
      result[i] = [];
      for (let j = 0; j < matrix.data[0].length; j++) {
        result[i][j] = 0;
        for (let k = 0; k < this.data[0].length; k++) {
          result[i][j] += this.data[i][k] * matrix.data[k][j];
        }
      }
    }
    return new Matrix(result);
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
      throw new Error("Invalid matrix size");
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
      throw new Error("Invalid matrix size");
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
}
