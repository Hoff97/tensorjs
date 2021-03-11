const webpackConfig = require('./webpack.config.js');

const webpack = require('webpack');

const path = require('path');

const process = require('process');
process.env.CHROME_BIN = require('puppeteer').executablePath();

const debug = process.argv.find(x => x === '--debug') !== undefined;

const webpackRules = [
  {
    test: /\.tsx?$/,
    use: 'ts-loader',
    exclude: /node_modules/,
  },
];
if (!debug) {
  webpackRules.push({
    test: /\.ts$/,
    exclude: [path.resolve(__dirname, 'test')],
    enforce: 'post',
    use: {
      loader: 'istanbul-instrumenter-loader',
      options: {esModules: true},
    },
  });
}

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
      '**/*.ts': ['webpack', 'sourcemap'],
    },
    mime: {'text/x-typescript': ['ts', 'tsx']},
    webpack: {
      module: {
        rules: webpackRules,
      },
      resolve: webpackConfig.resolve,
      mode: 'development',
      devtool: 'eval-source-map',
    },
    webpackMiddleware: {
      logLevel: 'info',
    },
    plugins: [
      'karma-chrome-launcher',
      'karma-jasmine',
      'karma-sourcemap-loader',
      'karma-webpack',
      'karma-coverage-istanbul-reporter',
      'karma-spec-reporter',
      'karma-firefox-launcher',
    ],
    reporters: debug ? ['spec'] : ['spec', 'coverage-istanbul'],
    port: 9876,
    colors: true,

    coverageIstanbulReporter: debug
      ? undefined
      : {
          reports: ['html', 'text-summary', 'lcovonly', 'json'],
          dir: path.join(__dirname, 'coverage'),
          fixWebpackSourcePaths: true,
          'report-config': {
            html: {outdir: 'html'},
          },
        },

    // level of logging
    // possible values: config.LOG_DISABLE || config.LOG_ERROR || config.LOG_WARN || config.LOG_INFO || config.LOG_DEBUG
    logLevel: config.LOG_INFO,

    // enable / disable watching file and executing tests whenever any file changes
    autoWatch: true,

    // When debugging, the browser shouldnt be timed out too quickly,
    // since one might stay in one breakpoint for a while
    browserNoActivityTimeout: debug ? 600 * 1000 : 30 * 1000,
    browsers: debug ? [] : ['ChromeDebugging'],

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
