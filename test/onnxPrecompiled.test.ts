import { OnnxModel } from '../lib/onnx/model';
import { toGPU } from '../lib/util/convert';

import { enabledTests } from './data/enabledTests';
import { onnx } from 'onnx-proto';
import { createTensor } from '../lib/onnx/util';
import Tensor from '../lib/types';

const run = false;

const epsilon = 0.001;

const opsetVersions = ['9'];

// These are not suitable since either their input is
// outside of the range of float16 or the errors add up too much
// TODO: Maybe overwrite the behavior of these operators
// to use float32 in any case?
const excludeHalfPrecision = new Set([
  'test_softmax_large_number',
  'test_upsample_nearest',
  'test_reduce_sum_square_do_not_keepdims_random',
  'test_reduce_sum_square_keepdims_random',
  'test_reduce_sum_square_default_axes_keepdims_random'
]);

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

        const result1 = (await model.forward(inputsDevice))[0];
        expect(await result1.compare(out, epsilon)).toBeTrue();
        const result2 = (await model.forward(inputsDevice))[0];
        expect(await result2.compare(out, epsilon)).toBeTrue();
      });

      if (!excludeHalfPrecision.has(test)) {
        it(`Should work for operator ${test} with half precision`, async () => {
          const resp = await fetch(`onnx/${opset}/${test}/model.onnx`);
          const buffer = await resp.arrayBuffer();

          const model = new OnnxModel(buffer, {
            precision: 16
          });

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
            inputsDevice.push(await toGPU(inputs[i], 16));
          }

          const result1 = (await model.forward(inputsDevice))[0];
          expect(await result1.copy().compare(out, epsilon*30)).toBeTrue();
          const result2 = (await model.forward(inputsDevice))[0];
          expect(await result2.copy().compare(out, epsilon*30)).toBeTrue();
        });
      }
    }
  });
}

}