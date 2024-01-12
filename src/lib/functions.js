"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.mean = exports.crossEntropyLoss = exports.sigmoid = void 0;
var sigmoid = function (input) {
  return 1 / (1 + Math.exp(-input));
};
exports.sigmoid = sigmoid;
var crossEntropyLoss = function (target, prediction) {
  return -(
    target * Math.log10(prediction) +
    (1 - target) * Math.log10(1 - prediction)
  );
};
exports.crossEntropyLoss = crossEntropyLoss;
var mean = function (numbers) {
  var sum = numbers.reduce(function (a, b) {
    return a + b;
  }, 0);
  return sum / numbers.length || 0;
};
exports.mean = mean;
