const inputs = [0.1, 0.5, 0.2];
const weights = [0.4, 0.3, 0.6];
const threshold = 0.5;

export const singleLayerPerceptron = () => {
  const weightedSum = inputs
    .map((num, index) => num * weights[index])
    .reduce((accumulator, currentValue) => accumulator + currentValue, 0);

  console.log(weightedSum);
  return stepFunction(weightedSum);
};

const stepFunction = (weightedSum: number) => {
  return weightedSum > threshold ? 1 : 0;
};
