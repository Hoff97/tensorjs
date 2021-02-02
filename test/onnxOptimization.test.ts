import {OnnxModel} from '../lib/onnx/model';
import {toCPU, toGPU, toWASM} from '../lib/util/convert';

import Tensor from '../lib/types';
import {getSize} from '../lib/util/shape';
import {CPUTensor} from '../lib/tensor/cpu/tensor';

const run = true;

const epsilon = 0.001;

const backends = ['CPU', 'WASM' /*, 'GPU'*/];

function randomValues(length: number) {
  const values: number[] = [];
  for (let i = 0; i < length; i++) {
    values.push(Math.random());
  }
  return values;
}

const tests = [
  {
    name: 'conv_batchnorm',
    inputsShape: [[1, 3, 6, 6]],
  },
  {
    name: 'conv_batchnorm_relu',
    inputsShape: [[1, 3, 6, 6]],
  },
  {
    name: 'conv_batchnorm_relu6',
    inputsShape: [[1, 3, 6, 6]],
  },
];

if (run) {
  for (const test of tests) {
    describe(`Onnx optimization ${test.name}`, () => {
      for (const backend of backends) {
        it(`Should work for backend ${backend}`, async () => {
          const resp = await fetch(`onnx/optimizations/${test.name}.onnx`);
          const buffer1 = await resp.arrayBuffer();

          const model1 = new OnnxModel(buffer1);
          const model2 = new OnnxModel(buffer1);

          const inputs: Tensor[] = [];
          for (const inputShape of test.inputsShape) {
            const values = randomValues(getSize(inputShape));
            inputs.push(new CPUTensor(inputShape, new Float32Array(values)));
          }

          if (backend === 'CPU') {
            await model1.toCPU();
            await model2.toCPU();
            for (let i = 0; i < inputs.length; i++) {
              inputs[i] = await toCPU(inputs[i]);
            }
          } else if (backend === 'GPU') {
            await model1.toGPU();
            await model2.toGPU();
            for (let i = 0; i < inputs.length; i++) {
              inputs[i] = await toGPU(inputs[i], 32);
            }
          } else {
            await model1.toWASM();
            await model2.toWASM();
            for (let i = 0; i < inputs.length; i++) {
              inputs[i] = await toWASM(inputs[i]);
            }
          }

          model1.optimize();

          const result1 = (await model1.forward(inputs))[0];
          const result2 = (await model2.forward(inputs))[0];
          expect(await result1.compare(result2, epsilon)).toBeTrue();
        });
      }
    });
  }
}
