import { rmdirSync, existsSync, mkdirSync } from 'fs'
import { execSync } from 'child_process';

import { enabledTests } from '../test/enabledTests';

const dataDir = 'test/data/onnx';

if (existsSync(dataDir)) {
  execSync(`rm -rf ${dataDir}`)
}
if (existsSync('tmp')) {
  execSync('rm -rf ./tmp')
}
mkdirSync(dataDir);

const opsetToTag = {
  '9': 'v1.4.0'
}

mkdirSync('./tmp');

for (let opset in opsetToTag) {
  const command = `cd tmp && git clone --depth 1 --branch ${opsetToTag[opset]} https://github.com/onnx/onnx.git`;
  execSync(command);

  const copyDir = `${dataDir}/${opset}`;

  mkdirSync(copyDir);

  for (let enabledTest of enabledTests) {
    console.log(enabledTest);
    execSync(`cp -r tmp/onnx/onnx/backend/test/data/node/${enabledTest} ${copyDir}/${enabledTest}`);
  }
  execSync('rm -rf tmp/onnx');
}

execSync('rm -rf ./tmp')