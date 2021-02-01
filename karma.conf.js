const webpackConfig = require('./webpack.config.js');

const webpack = require('webpack');

const path = require('path');

const process = require('process');
process.env.CHROME_BIN = require('puppeteer').executablePath();

module.exports = function (config) {
  config.set({
    basePath: '',
    frameworks: ['jasmine'],
    files: [
      'test/*.ts',
      {
        pattern: 'test/data/onnx/**',
        included: false,
        served: true,
        watched: false,
        nocache: true,
      },
    ],
    proxies: {
      '/onnx/': 'http://localhost:9876/base/test/data/onnx/',
    },
    exclude: [],
    preprocessors: {
      '**/*.ts': ['webpack'],
    },
    mime: {'text/x-typescript': ['ts', 'tsx']},
    webpack: {
      module: webpackConfig.module,
      resolve: webpackConfig.resolve,
      mode: 'development',
      devtool: false,
      plugins: [
        new webpack.SourceMapDevToolPlugin({
          test: /\.(ts|js|css)($|\?)/i,
          filename: null,
        }),
      ],
    },
    webpackMiddleware: {
      logLevel: 'error',
    },
    reporters: ['spec'],
    port: 9876,
    colors: true,

    // level of logging
    // possible values: config.LOG_DISABLE || config.LOG_ERROR || config.LOG_WARN || config.LOG_INFO || config.LOG_DEBUG
    logLevel: config.LOG_INFO,

    // enable / disable watching file and executing tests whenever any file changes
    autoWatch: true,

    browsers: ['ChromeDebugging'],

    customLaunchers: {
      ChromeDebugging: {
        base: 'ChromeHeadless',
        flags: ['--remote-debugging-port=9333'],
      },
      ChromeGPU: {
        base: 'Chrome',
        flags: [
          '--use-gl=desktop',
          '--enable-webgl',
          '--disable-swiftshader',
          '--remote-debugging-port=9333',
        ],
      },
    },

    // Continuous Integration mode
    // if true, Karma captures browsers, runs the tests and exits
    singleRun: process.env.CI ? true : false,
    concurrency: 1,
    client: {
      captureConsole: true,
    },
  });
};
