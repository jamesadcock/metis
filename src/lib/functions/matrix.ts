export class Matrix {
  private data: number[][];

  constructor(data: number[][]) {
    this.data = data;
  }

  get() {
    return this.data;
  }

  public multiply(matrix: Matrix): number[][] {
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
    return result;
  }

  public transpose(): number[][] {
    const result: number[][] = [];
    for (let i = 0; i < this.data[0].length; i++) {
      result[i] = [];
      for (let j = 0; j < this.data.length; j++) {
        result[i][j] = this.data[j][i];
      }
    }
    return result;
  }
}
