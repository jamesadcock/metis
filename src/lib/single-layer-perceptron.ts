import { sigmoid, crossEntropyLoss as lossFunction, mean } from "./functions";

export interface SingleLayerPerceptronProps {
  bias: number;
  features: Feature[];
  weights: number[];
  learningRate: number;

  epochs: number;
}

export interface Feature {
  params: number[];
  target: number;
}

export const singleLayerPerceptron = ({
  bias,
  features,
  weights,
  learningRate,
  epochs,
}: SingleLayerPerceptronProps) => {
  let i = 0;
  let averageLoss: number;
  while (i < epochs) {
    const loss = features.map((feature) => {
      const pred = predict(feature, weights, bias);
      const distance = lossFunction(feature.target, pred);

      weights = weights.map((weight, index) => {
        return updateWeight(
          weight,
          pred,
          feature.params[index],
          learningRate,
          feature.target,
        );
      });

      bias = updateBias(bias, pred, learningRate, feature.target);
      return distance;
    });

    averageLoss = mean(loss);
    i++;
  }

  return { averageLoss, weights, bias };
};

const updateWeight = (
  oldWeight: number,
  prediction: number,
  input: number,
  learningRate: number,
  target: number,
) => {
  return oldWeight + learningRate * (target - prediction) * input;
};

const updateBias = (
  oldBias: number,
  prediction: number,
  learningRate: number,
  target: number,
) => {
  return oldBias + learningRate * (target - prediction);
};

const predict = (feature: Feature, weights: number[], bias: number) => {
  const weightedSum =
    feature.params
      .map((num, index) => num * weights[index])
      .reduce((accumulator, currentValue) => accumulator + currentValue, 0) +
    bias;

  return sigmoid(weightedSum);
};
