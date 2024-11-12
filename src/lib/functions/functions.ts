export const sigmoid = (input: number): number => {
  return 1 / (1 + Math.exp(-input));
};

export const crossEntropyLoss = (
  target: number,
  prediction: number,
): number => {
  return -(
    target * Math.log(prediction) +
    (1 - target) * Math.log(1 - prediction)
  );
};

export const mean = (numbers: number[]) => {
  const sum = numbers.reduce((a, b) => a + b, 0);
  return sum / numbers.length || 0;
};
