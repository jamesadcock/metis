const fs = require("fs");
const path = require("path");

function readCSVFile(filePath) {
  try {
    // Read the file content
    const fileContent = fs.readFileSync(filePath, "utf8");

    // Split the content into rows
    const rows = fileContent.split("\n");

    // Process each row into an object with 'params' and 'target'
    const result = rows.map((row) => {
      const values = row.split(",").map((value) => {
        const number = parseFloat(value);
        return isNaN(number) ? value : number;
      });

      // Separate the last value as 'target' and the rest as 'params'
      return {
        params: values.slice(0, -1),
        target: values[values.length - 1],
      };
    });

    return result;
  } catch (error) {
    console.error("Error reading the CSV file:", error);
    return [];
  }
}

function writeJSONFile(data, filePath) {
  try {
    // Convert the data to JSON format
    const jsonData = JSON.stringify(data, null, 2);

    // Write the JSON data to file
    fs.writeFileSync(filePath, jsonData, "utf8");
    console.log("Data successfully written to", filePath);
  } catch (error) {
    console.error("Error writing the JSON file:", error);
  }
}

// Replace 'path/to/your/file.csv' with the path to your CSV file
const csvFilePath = path.join(__dirname, "data.csv");
const jsonFilePath = path.join(__dirname, "data.json");

const data = readCSVFile(csvFilePath);
writeJSONFile(data, jsonFilePath);
