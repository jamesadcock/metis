export const sigmoid = (input: number) => {
  console.log(input);
  return 1 / (1 + Math.exp(-input));
};
