module.exports = {
  preset: "ts-jest",
  transform: { "^.+\\.tsx?$": ["ts-jest", { isolatedModules: true }] },
  transformIgnorePatterns: ["<rootDir>/node_modules/"],
};
