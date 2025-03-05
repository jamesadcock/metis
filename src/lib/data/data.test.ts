const mockWriteFileSync = jest.fn();
jest.mock("fs", () => ({
  ...jest.requireActual("fs"),
  writeFileSync: mockWriteFileSync,
}));

import { Data } from "./data";

describe("loadTraining", () => {
  it(`should return the correct data as a single batch when no batch
      size is provided`, () => {
    const result = Data.loadTraining("test-data/pizza-small.csv");
    expect(result.trainingFeatures[0].get()).toStrictEqual([
      [13, 26, 9],
      [2, 14, 6],
    ]);
    expect(result.trainingLabels[0].get()).toStrictEqual([[44], [23]]);
    expect(result.batchSize).toBe(0);
    expect(result.lastBatchSize).toBe(2);
    expect(result.numberOfBatches).toBe(1);
  });

  it(`should return the correct data with 2 batches when a batch
    size of 1 provided`, () => {
    const result = Data.loadTraining("test-data/pizza-small.csv", 1);
    expect(result.trainingFeatures[0].get()).toStrictEqual([[13, 26, 9]]);
    expect(result.trainingFeatures[1].get()).toStrictEqual([[2, 14, 6]]);

    expect(result.trainingLabels[0].get()).toStrictEqual([[44]]);
    expect(result.trainingLabels[1].get()).toStrictEqual([[23]]);

    expect(result.batchSize).toBe(1);
    expect(result.lastBatchSize).toBe(0);
    expect(result.numberOfBatches).toBe(2);
  });

  it("should throw an error", () => {
    expect(() => {
      Data.loadTraining("test-data/missing-file.csv");
    }).toThrow("Error reading the CSV file");
  });
});

describe("load", () => {
  it("should return the correct data", () => {
    const result = Data.loadValidationAndTest("test-data/iris-test.csv");
    expect(result.testFeatures.get()).toStrictEqual([
      [5, 3, 1.6, 0.2],
      [5.7, 2.8, 4.1, 1.3],
    ]);

    expect(result.validationFeatures.get()).toStrictEqual([
      [5.2, 4.1, 1.5, 0.1],
      [5.1, 2.5, 3, 1.1],
    ]);

    expect(result.testLabels.get()).toStrictEqual([[0], [1]]);
    expect(result.validationLabels.get()).toStrictEqual([[0], [1]]);
  });

  it("should throw an error", () => {
    expect(() => {
      Data.loadValidationAndTest("test-data/missing-file.csv");
    }).toThrow("Error reading the CSV file");
  });
});
