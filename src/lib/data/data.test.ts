const mockWriteFileSync = jest.fn();
jest.mock("fs", () => ({
  ...jest.requireActual("fs"),
  writeFileSync: mockWriteFileSync,
}));

import { Data } from "./data";

describe("loadTraining", () => {
  it("should return the correct data", () => {
    const result = Data.loadTraining("test-data/pizza-small.csv");
    expect(result.features.get()).toStrictEqual([
      [13, 26, 9],
      [2, 14, 6],
    ]);
    expect(result.labels.get()).toStrictEqual([[44], [23]]);
  });

  it("should throw an error", () => {
    expect(() => {
      Data.loadTraining("test-data/missing-file.csv");
    }).toThrow("Error reading the CSV file");
  });
});

describe("load", () => {
  it("should return the correct data", () => {
    const result = Data.load("test-data/iris-test.csv");
    expect(result.data.get()).toStrictEqual([
      [5, 3, 1.6, 0.2],
      [5.7, 2.8, 4.1, 1.3],
      [5.2, 4.1, 1.5, 0.1],
    ]);
  });

  it("should throw an error", () => {
    expect(() => {
      Data.load("test-data/missing-file.csv");
    }).toThrow("Error reading the CSV file");
  });
});
