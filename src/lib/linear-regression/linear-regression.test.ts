import { Data } from "../data/data";
import { Matrix } from "../functions/matrix";
import { LinearRegression } from "./linear-regression";

describe("linear-regression", () => {
  it("should return the correct data", () => {
    const trainingData = Data.loadTraining("test-data/pizza.csv");
    const linearRegression = new LinearRegression();
    const weights = linearRegression.train(
      trainingData.features,
      trainingData.labels,
      0.00005,
      100000,
    );
    const prediction1Data = new Matrix([[6, 14, 9]]);
    const prediction2Data = new Matrix([[13, 19, 4]]);
    const prediction3Data = new Matrix([[13, 20, 3]]);
    const prediction1 = linearRegression.predict(prediction1Data, weights);
    const prediction2 = linearRegression.predict(prediction2Data, weights);
    const prediction3 = linearRegression.predict(prediction3Data, weights);

    expect(prediction1.get()[0][0] - 38).toBeLessThan(5);
    expect(prediction2.get()[0][0] - 30).toBeLessThan(5);
    expect(prediction3.get()[0][0] - 28).toBeLessThan(5);
  });
});
