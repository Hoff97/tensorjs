import { CPUTensor } from '../lib/tensor/cpu/tensor';
import { OnnxModel } from '../lib/onnx/model';
import { toCPU, toGPU, toWASM } from '../lib/util/convert';

import { enabledTests } from './enabledTests';
import { onnx } from 'onnx-proto';
import { createTensor } from '../lib/onnx/util';
import Tensor from '../lib/types';

/*const b: ArrayBuffer = require('arraybuffer-loader!./data/test.onnx');

describe(`Onnx model`, () => {
  it('should be able to load a MobileNet', async () => {
    const model = new OnnxModel(b);

    await model.toGPU();

    const input = await toGPU(new CPUTensor([1,3,224,224]));
    const result = model.forward([input]);

    const sM = result[0].softmax(1);

    const values = await sM.getValues();
  })
});*/

const epsilon = 0.00001;

const opsetVersions = ['9'];

const backends = ['CPU', 'WASM', 'GPU'];

for (let opset of opsetVersions) {
  describe(`Opset ${opset}`, () => {
    for (let test of enabledTests) {
      for (let backend of backends) {
        it(`Should work for operator ${test} with backend ${backend}`, async () => {
          const resp = await fetch(`onnx/${opset}/${test}/model.onnx`);
          const buffer = await resp.arrayBuffer();

          const model = new OnnxModel(buffer);

          const inputs: Tensor[] = [];
          let i = 0;
          while (true) {
            const resp = await fetch(`onnx/${opset}/${test}/test_data_set_0/input_${i}.pb`);
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

          const respOut = await fetch(`onnx/${opset}/${test}/test_data_set_0/output_0.pb`);
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
            out = await toGPU(output);
            for (let i = 0; i < inputs.length; i++) {
              inputsDevice.push(await toGPU(inputs[i]));
            }
          } else {
            model.toWASM();
            out = await toWASM(output);
            for (let i = 0; i < inputs.length; i++) {
              inputsDevice.push(await toWASM(inputs[i]));
            }
          }

          const result = model.forward(inputsDevice)[0];

          expect(await result.compare(out, epsilon)).toBeTruthy();
        });
      }
    }
  });
}