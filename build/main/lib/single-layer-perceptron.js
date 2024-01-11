"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.singleLayerPerceptron = void 0;
const inputs = [0.1, 0.5, 0.2];
const weights = [0.4, 0.3, 0.6];
const threshold = 0.5;
const singleLayerPerceptron = () => {
    const weightedSum = inputs
        .map((num, index) => num * weights[index])
        .reduce((accumulator, currentValue) => accumulator + currentValue, 0);
    console.log(weightedSum);
    return stepFunction(weightedSum);
};
exports.singleLayerPerceptron = singleLayerPerceptron;
const stepFunction = (weightedSum) => {
    return weightedSum > threshold ? 1 : 0;
};
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoic2luZ2xlLWxheWVyLXBlcmNlcHRyb24uanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi9zcmMvbGliL3NpbmdsZS1sYXllci1wZXJjZXB0cm9uLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiI7OztBQUFBLE1BQU0sTUFBTSxHQUFHLENBQUMsR0FBRyxFQUFFLEdBQUcsRUFBRSxHQUFHLENBQUMsQ0FBQztBQUMvQixNQUFNLE9BQU8sR0FBRyxDQUFDLEdBQUcsRUFBRSxHQUFHLEVBQUUsR0FBRyxDQUFDLENBQUM7QUFDaEMsTUFBTSxTQUFTLEdBQUcsR0FBRyxDQUFDO0FBRWYsTUFBTSxxQkFBcUIsR0FBRyxHQUFHLEVBQUU7SUFDeEMsTUFBTSxXQUFXLEdBQUcsTUFBTTtTQUN2QixHQUFHLENBQUMsQ0FBQyxHQUFHLEVBQUUsS0FBSyxFQUFFLEVBQUUsQ0FBQyxHQUFHLEdBQUcsT0FBTyxDQUFDLEtBQUssQ0FBQyxDQUFDO1NBQ3pDLE1BQU0sQ0FBQyxDQUFDLFdBQVcsRUFBRSxZQUFZLEVBQUUsRUFBRSxDQUFDLFdBQVcsR0FBRyxZQUFZLEVBQUUsQ0FBQyxDQUFDLENBQUM7SUFFeEUsT0FBTyxDQUFDLEdBQUcsQ0FBQyxXQUFXLENBQUMsQ0FBQztJQUN6QixPQUFPLFlBQVksQ0FBQyxXQUFXLENBQUMsQ0FBQztBQUNuQyxDQUFDLENBQUM7QUFQVyxRQUFBLHFCQUFxQix5QkFPaEM7QUFFRixNQUFNLFlBQVksR0FBRyxDQUFDLFdBQW1CLEVBQUUsRUFBRTtJQUMzQyxPQUFPLFdBQVcsR0FBRyxTQUFTLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO0FBQ3pDLENBQUMsQ0FBQyJ9