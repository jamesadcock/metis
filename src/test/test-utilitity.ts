const math = global.Math;

export const mockMathRandom = (value: number) => {
  const mockMath = Object.create(global.Math);
  mockMath.random = () => value;
  global.Math = mockMath;
};

export const resetMockMathRandom = () => {
  global.Math = math;
};

export const roundMatrix = (
  matrix: number[][],
  decimalPlaces: number,
): number[][] => {
  return matrix.map((row) => {
    return row.map((num) => {
      const multiplier = Math.pow(10, decimalPlaces);
      return Math.round(num * multiplier) / multiplier;
    });
  });
};
