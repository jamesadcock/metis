import { roundNumber } from "./test-utilitity";

describe("roundNumber", () => {
  it("should round number down", () => {
    const result = roundNumber(0.123456789, 2);
    expect(result).toEqual(0.12);
  });

  it("should round number up", () => {
    const result = roundNumber(0.125, 2);
    expect(result).toEqual(0.13);
  });
});
