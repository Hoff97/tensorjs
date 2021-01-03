const webpackConfig = require('./webpack.config.js');

const process = require('process');
process.env.CHROME_BIN = require('puppeteer').executablePath();

module.exports = function(config) {
    config.set({
        basePath: '',
        frameworks: ['benchmark'],
        files: ['test/**/performance/*.ts'],
        exclude: [],
        preprocessors: {
            'test/**/performance/*.ts': ['webpack']
        },
        webpack: {
            module: webpackConfig.module,
            resolve: webpackConfig.resolve,
            mode: webpackConfig.mode,
            devtool: 'inline-source-map',
        },
        reporters: ['benchmark'],
        port: 9876,
        colors: true,

        // possible values: config.LOG_DISABLE || config.LOG_ERROR || config.LOG_WARN || config.LOG_INFO || config.LOG_DEBUG
        logLevel: config.LOG_INFO,
        autoWatch: false,
        browsers: [
            'ChromeGPU'
        ],
        
        customLaunchers: {
            ChromeGPU: {
                base: 'Chrome',
                flags: ['--use-gl=desktop', '--enable-webgl', '--disable-swiftshader']
            }
        },

        singleRun: true,
        concurrency: 1,
        client: {
            captureConsole: true,
        }
    });
};