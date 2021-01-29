import { OnnxModel } from '../lib/onnx/model';
import { toGPU } from '../lib/util/convert';

import { enabledTests } from './data/enabledTests';
import { onnx } from 'onnx-proto';
import { createTensor } from '../lib/onnx/util';
import Tensor from '../lib/types';

const run = false;

const epsilon = 0.001;

const opsetVersions = ['9'];

if (run) {
for (let opset of opsetVersions) {
  describe(`Opset ${opset} precompiled`, () => {
    for (let test of enabledTests) {
      it(`Should work for operator ${test}`, async () => {
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

        await model.toGPU();
        out = await toGPU(output, 32);
        for (let i = 0; i < inputs.length; i++) {
          inputsDevice.push(await toGPU(inputs[i], 32));
        }

        await model.compileForInputs(inputsDevice);

        const result1 = (await model.forward(inputsDevice))[0];
        expect(await result1.compare(out, epsilon)).toBeTrue();
        const result2 = (await model.forward(inputsDevice))[0];
        expect(await result2.compare(out, epsilon)).toBeTrue();
      });
    }
  });
}

}