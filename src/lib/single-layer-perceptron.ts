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

export class SingleLayerPerceptron {
  private bias: number;
  private features: Feature[];
  private weights: number[];
  private learningRate: number;
  private epochs: number;
  private isTrained: boolean = false;

  constructor({
    bias,
    features,
    weights,
    learningRate,
    epochs,
  }: SingleLayerPerceptronProps) {
    this.bias = bias;
    this.features = features;
    this.weights = weights;
    this.learningRate = learningRate;
    this.epochs = epochs;
  }

  public train() {
    let i = 0;
    let averageLoss: number;
    while (i < this.epochs) {
      const loss = this.features.map((feature) => {
        const pred = this.predict(feature);
        const distance = lossFunction(feature.target, pred);

        this.weights = this.weights.map((weight, index) => {
          return this.updateWeight(
            weight,
            pred,
            feature.params[index],
            this.learningRate,
            feature.target,
          );
        });

        this.bias = this.updateBias(
          this.bias,
          pred,
          this.learningRate,
          feature.target,
        );
        return distance;
      });

      averageLoss = mean(loss);
      i++;
    }

    this.isTrained = true;
    return { averageLoss, weights: this.weights, bias: this.bias };
  }

  public predictFeature(feature: Feature) {
    if (!this.isTrained) {
      throw new Error("Model is not trained yet");
    }

    const result = this.predict(feature);
    return result > 0.5 ? 1 : 0;
  }

  private predict(feature: Feature) {
    const weightedSum =
      feature.params
        .map((num, index) => num * this.weights[index])
        .reduce((accumulator, currentValue) => accumulator + currentValue, 0) +
      this.bias;

    return sigmoid(weightedSum);
  }

  private updateWeight = (
    weight: number,
    prediction: number,
    feature: number,
    learningRate: number,
    target: number,
  ) => {
    return weight + learningRate * (target - prediction) * feature;
  };

  private updateBias = (
    bias: number,
    prediction: number,
    learningRate: number,
    target: number,
  ) => {
    return bias + learningRate * (target - prediction);
  };
}
