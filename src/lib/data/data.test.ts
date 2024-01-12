const mockWriteFileSync = jest.fn();
jest.mock("fs", () => ({
  ...jest.requireActual("fs"),
  writeFileSync: mockWriteFileSync,
}));

import { Data } from "./data";

describe("data", () => {
  it("should return the correct data", () => {
    Data.convertCsvToTrainingFeature("test-data/iris-binary-data-small.csv");
    expect(mockWriteFileSync).toHaveBeenCalledWith(
      "test-data/iris-binary-data-small.csv",
      JSON.stringify(
        [
          { params: [5, 3, 1.6, 0.2], target: 0 },
          { params: [5.7, 2.8, 4.1, 1.3], target: 1 },
          { params: [5.2, 4.1, 1.5, 0.1], target: 0 },
        ],
        null,
        2
      ),
      "utf8"
    );
  });
});
