import { splitArrayInHalf } from "./utility";

describe("splitArrayInHalf", () => {
  it("should correctly split the data", () => {
    const data = [[1], [2], [3], [4], [5]];
    const result = splitArrayInHalf(data);
    expect(result.firstHalf).toStrictEqual([[1], [2], [3]]);
    expect(result.secondHalf).toStrictEqual([[4], [5]]);
  });
});
