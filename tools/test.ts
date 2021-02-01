import {existsSync, mkdirSync} from 'fs';
import {execSync} from 'child_process';

import {enabledTests} from '../test/data/enabledTests';
import {models} from '../test/data/models';

const dataDir = 'test/data/onnx';

function loadOnnxUnitTests() {
  const opsetToTag = {
    '9': 'v1.4.0',
  };

  for (const opset in opsetToTag) {
    const command = `cd tmp && git clone --depth 1 --branch ${opsetToTag[opset]} https://github.com/onnx/onnx.git`;
    execSync(command);

    const copyDir = `${dataDir}/${opset}`;

    mkdirSync(copyDir);

    for (const enabledTest of enabledTests) {
      console.log(enabledTest);
      execSync(
        `cp -r tmp/onnx/onnx/backend/test/data/node/${enabledTest} ${copyDir}/${enabledTest}`
      );
    }
    execSync('rm -rf tmp/onnx');
  }
}

function loadOnnxModels() {
  mkdirSync(`${dataDir}/models`);

  for (const model of models) {
    const command = `cd tmp && wget ${model.url}`;
    execSync(command);

    if (model.zipped) {
      execSync(`cp tmp/${model.fileName} ${dataDir}/models`);
      execSync(`cd ${dataDir}/models && tar -xvzf ${model.fileName}`);
    } else {
      const copyDir = `${dataDir}/models/${model.name}`;
      mkdirSync(copyDir);
      execSync(`cp tmp/${model.fileName} ${copyDir}`);
      execSync(`cd ${copyDir} && mv ${model.fileName} ${model.name}.onnx`)
    }
  }
}

if (existsSync(dataDir)) {
  execSync(`rm -rf ${dataDir}`);
}
if (existsSync('tmp')) {
  execSync('rm -rf ./tmp');
}
mkdirSync(dataDir);

mkdirSync('./tmp');

loadOnnxUnitTests();
loadOnnxModels();

execSync('rm -rf ./tmp');
