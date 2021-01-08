const path = require("path");
const CopyPlugin = require("copy-webpack-plugin");
const WasmPackPlugin = require("@wasm-tool/wasm-pack-plugin");

const dist = path.resolve(__dirname, "dist");

module.exports = env => {
  return {
    mode: env === "production" ? "production" : "development",
    entry: {
      index: "./lib/library.ts"
    },
    output: {
      path: dist,
      filename: "[name].js"
    },
    module: {
      rules: [
        {
          test: /\.tsx?$/,
          use: 'ts-loader',
          exclude: /node_modules/,
        },
      ],
    },
    resolve: {
      extensions: [ '.tsx', '.ts', '.js' ],
    },
    output: {
      filename: 'bundle.js',
      path: path.resolve(__dirname, 'dist'),
    },
    plugins: [
      new WasmPackPlugin({
        crateDirectory: __dirname,
      }),
    ]
  }
};
