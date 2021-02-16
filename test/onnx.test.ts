import {OnnxModel} from '../lib/onnx/model';
import {toCPU, toGPU, toWASM} from '../lib/util/convert';

import {enabledTests, opsetVersions} from './data/enabledTests';
import {onnx} from 'onnx-proto';
import {createTensor} from '../lib/onnx/util';
import Tensor from '../lib/types';

const run = true;

const epsilon = 0.001;

const backends = ['CPU', 'WASM' /*, 'GPU'*/];

if (run) {
  for (const opset of opsetVersions) {
    describe(`Opset ${opset}`, () => {
      for (const test of enabledTests) {
        const testName = typeof test === 'string' ? test : test.name;

        const runForOpset =
          typeof test === 'string' ||
          test.opsets === undefined ||
          test.opsets.find(os => opset === os) !== undefined;

        if (runForOpset) {
          for (const backend of backends) {
            it(`Should work for operator ${testName} with backend ${backend}`, async () => {
              const resp = await fetch(`onnx/${opset}/${testName}/model.onnx`);
              const buffer = await resp.arrayBuffer();

              const model = new OnnxModel(buffer);

              const inputs: Tensor[] = [];
              let i = 0;
              // eslint-disable-next-line no-constant-condition
              while (true) {
                const resp = await fetch(
                  `onnx/${opset}/${testName}/test_data_set_0/input_${i}.pb`
                );
                if (resp.status !== 200) {
                  break;
                }
                const buffer = await resp.arrayBuffer();
                const arr = new Uint8Array(buffer);
                const tensorProto = onnx.TensorProto.decode(arr);
                const tensor = createTensor(tensorProto);
                inputs.push(tensor);
                i++;
              }

              const respOut = await fetch(
                `onnx/${opset}/${testName}/test_data_set_0/output_0.pb`
              );
              const bufferOut = await respOut.arrayBuffer();
              const arr = new Uint8Array(bufferOut);
              const tensorProto = onnx.TensorProto.decode(arr);
              const output = createTensor(tensorProto);

              let out: Tensor;
              const inputsDevice: Tensor[] = [];
              if (backend === 'CPU') {
                await model.toCPU();
                out = await toCPU(output);
                for (let i = 0; i < inputs.length; i++) {
                  inputsDevice.push(await toCPU(inputs[i]));
                }
              } else if (backend === 'GPU') {
                await model.toGPU();
                out = await toGPU(output, 32);
                for (let i = 0; i < inputs.length; i++) {
                  inputsDevice.push(await toGPU(inputs[i], 32));
                }
              } else {
                model.toWASM();
                out = await toWASM(output);
                for (let i = 0; i < inputs.length; i++) {
                  inputsDevice.push(await toWASM(inputs[i]));
                }
              }

              const result1 = (await model.forward(inputsDevice))[0];
              expect(await result1.compare(out, epsilon)).toBeTrue();
              const result2 = (await model.forward(inputsDevice))[0];
              expect(await result2.compare(out, epsilon)).toBeTrue();
            });
          }
        }
      }
    });
  }
}
