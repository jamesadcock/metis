import * as http from "http";
import * as fs from "fs";
import { NeuralNetwork } from "./lib/neural-network/neural-network";
import { Mnist } from "./lib/data/mnist";
import { Matrix } from "./lib/functions/matrix";

const port = 3000;

const runServer = () => {
  const server = http.createServer((req, res) => {
    console.log(req.url);
    if (req.url === "/digit.png") {
      // Serve the chart image
      fs.readFile("./digit.png", (err, data) => {
        if (err) {
          res.writeHead(404, { "Content-Type": "text/plain" });
          res.end("Number not found");
        } else {
          res.writeHead(200, {
            "Content-Type": "image/png",
            "Cache-Control": "no-store",
          });
          res.end(data);
        }
      });
    } else {
      // Serve the HTML page
      const image = renderImage();
      const num = predictNumber(image);
      const htmlContent = generateHTML("Identify the number", num);
      res.writeHead(200, { "Content-Type": "text/html" });
      res.end(htmlContent);
    }
  });

  // Start the server
  server.listen(port, () => {
    console.log(`Server is running at http://localhost:${port}`);
  });
};

const randomIndex = (maxNumber: number): number => {
  return Math.floor(Math.random() * maxNumber);
};

const renderImage = () => {
  const mnist = new Mnist();
  const images = mnist.loadImages("test-data/mnist/t10k-images-idx3-ubyte");
  const image = images[randomIndex(images.length)];
  mnist.renderImage(image);
  return image;
};

const predictNumber = (image: number[]): number => {
  const neuralNet = new NeuralNetwork();
  const imageMatrix = new Matrix([image]);

  // Load the weights
  const weights1 = JSON.parse(fs.readFileSync("weights1.json", "utf8"));
  const weights2 = JSON.parse(fs.readFileSync("weights2.json", "utf8"));

  // Classify the image
  const result = neuralNet.classify(imageMatrix, weights1, weights2);
  const num = result.get()[0][0];
  return num;
};

const generateHTML = (title: string, num: number): string => {
  return `
          <!DOCTYPE html>
          <html lang="en">
          <head>
              <meta charset="UTF-8">
              <meta name=""viewport" content="width=device-width, initial-scale=1.0">
          </head>
          <body style="background-color: #000000; color: #ffffff; text-align: center; padding-top: 100px; font-family: Arial, sans-serif;">
              <h1 style="font-size: 40px">${title}</h1>
              <img src="digit.png" alt="Chart height="128px" width="128px" />
              <div style="margin-top: 40px; font-size: 30px; color: #00ff00;">
                  Number recognised: <span id="number">${num}</span>
              </div>
              <button style="margin-top: 20px; padding: 10px 20px; font-size: 20px; cursor: pointer;" onclick="location.reload();">
                Next Number
              </button>
          </body>
          </html>
      `;
};

runServer();
