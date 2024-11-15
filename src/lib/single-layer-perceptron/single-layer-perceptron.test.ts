import { Data } from "../data/data";
import { SingleLayerPerceptron } from "./single-layer-perceptron";

describe("train", () => {
  it("should correctly predict", () => {
    const data = Data.convertCsvToMatrices("test-data/police.csv");
    const perceptron = new SingleLayerPerceptron();
    const weights = perceptron.train(data.features, data.labels, 0.001, 100);
    const results = perceptron.classify(data.features, weights);

    let correct = 0;
    let incorrect = 0;

    results.get().forEach((result, i) => {
      if (result[0] === data.labels.get()[i][0]) {
        correct++;
      } else {
        incorrect++;
      }
    });
    console.log("Correct: ", correct, "Incorrect: ", incorrect);
    expect(correct).toBe(26);
  });
});
