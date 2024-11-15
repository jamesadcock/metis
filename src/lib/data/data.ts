import fs from "fs";
import { Matrix } from "../functions/matrix";
export class Data {
  public static loadTraining(filePath: string) {
    try {
      const fileContent = fs.readFileSync(filePath, "utf8");
      const rows = fileContent.split("\n");
      const features: number[][] = [];
      const labels: number[][] = [];

      rows.map((row) => {
        const values = row.split(",").map((value) => {
          const number = parseFloat(value);
          return number;
        });

        features.push(values.slice(0, -1));
        labels.push([values[values.length - 1]]);
      });

      const featuresMatrix = new Matrix(features);
      const labelsMatrix = new Matrix(labels);

      return { features: featuresMatrix, labels: labelsMatrix };
    } catch (error) {
      throw new Error("Error reading the CSV file");
    }
  }

  public static load(filePath: string) {
    try {
      const fileContent = fs.readFileSync(filePath, "utf8");
      const rows = fileContent.split("\n");
      const data: number[][] = [];

      rows.map((row) => {
        const values = row.split(",").map((value) => {
          const number = parseFloat(value);
          return number;
        });

        data.push(values);
      });

      return { data: new Matrix(data) };
    } catch (error) {
      throw new Error("Error reading the CSV file");
    }
  }
}
