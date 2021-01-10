const webpackConfig = require('./webpack.config.js');

const process = require('process');
process.env.CHROME_BIN = require('puppeteer').executablePath();

module.exports = function(config) {
    config.set({
        basePath: '',
        frameworks: ['jasmine'],
        files: ['test/*.ts', {
            pattern: 'test/data/onnx/**',
            included: false,
            served: true,
            watched: false,
            nocache: true
        }],
        proxies: {
            "/onnx/": "http://localhost:9876/base/test/data/onnx/"
        },
        exclude: [],
        preprocessors: {
            'test/**/*.ts': ['webpack']
        },
        webpack: {
            module: webpackConfig.module,
            resolve: webpackConfig.resolve,
            mode: webpackConfig.mode,
            devtool: 'inline-source-map',
        },
        reporters: ['spec'],
        port: 9876,
        colors: true,

        // level of logging
        // possible values: config.LOG_DISABLE || config.LOG_ERROR || config.LOG_WARN || config.LOG_INFO || config.LOG_DEBUG
        logLevel: config.LOG_INFO,

        // enable / disable watching file and executing tests whenever any file changes
        autoWatch: true,

        browsers: [
            'ChromeDebugging'
        ],

        customLaunchers: {
            ChromeDebugging: {
                base: 'ChromeHeadless',
                flags: ['--remote-debugging-port=9333']
            },
            ChromeGPU: {
                base: 'Chrome',
                flags: ['--use-gl=desktop', '--enable-webgl', '--disable-swiftshader']
            }
        },

        // Continuous Integration mode
        // if true, Karma captures browsers, runs the tests and exits
        singleRun: process.env.CI ? true : false,
        concurrency: 1,
        client: {
            captureConsole: true,
        }
    });
};