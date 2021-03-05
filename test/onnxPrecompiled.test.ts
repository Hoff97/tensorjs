import {OnnxModel} from '../lib/onnx/model';
import {toGPU} from '../lib/util/convert';

import {enabledTests, opsetVersions} from './data/enabledTests';
import {onnx} from 'onnx-proto';
import {createTensor} from '../lib/onnx/util';
import Tensor from '../lib/types';
import {GPUTensor} from '../lib/tensor/gpu/tensor';

const run = false;

const epsilon = 0.001;

// These are not suitable since either their input is
// outside of the range of float16 or the errors add up too much
// TODO: Maybe overwrite the behavior of these operators
// to use float32 in any case?
const excludeHalfPrecision = new Set([
  'test_softmax_large_number',
  'test_upsample_nearest',
  'test_reduce_sum_square_do_not_keepdims_random',
  'test_reduce_sum_square_keepdims_random',
  'test_reduce_sum_square_default_axes_keepdims_random',
  'test_shape_example',
  'test_shape',
  'test_constantofshape_float_ones',
  'test_tan',
  'test_reduce_prod_default_axes_keepdims_random',
  'test_size_example',
  'test_size',
  'test_pow',
  'test_reduce_prod_default_axes_keepdims_example',
  'test_reduce_prod_keepdims_random',
  'test_reduce_prod_do_not_keepdims_random',
  'test_reduce_prod_keepdims_random',
]);

if (run) {
  for (const opset of opsetVersions) {
    describe(`Opset ${opset} precompiled`, () => {
      for (const test of enabledTests) {
        const testName = typeof test === 'string' ? test : test.name;

        const runForOpset =
          typeof test === 'string' ||
          test.opsets === undefined ||
          test.opsets.find(os => opset === os) !== undefined;

        const runForGPU =
          typeof test === 'string' ||
          test.backends === undefined ||
          test.backends.find(b => b === 'GPU') !== undefined;

        if (!excludeHalfPrecision.has(testName) && runForOpset && runForGPU) {
          it(`Should work for operator ${testName} with half precision`, async () => {
            const resp = await fetch(`onnx/${opset}/${testName}/model.onnx`);
            const buffer = await resp.arrayBuffer();

            // TODO: Adjust model loading!
            const model = new OnnxModel(buffer);

            const inputs: Tensor<'float32'>[] = [];
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

            const inputsDevice: Tensor<'float32'>[] = [];

            await model.toGPU();
            const out = await toGPU(output);
            for (let i = 0; i < inputs.length; i++) {
              inputsDevice.push(await toGPU(inputs[i]));
            }

            const result1 = (await model.forward(inputsDevice))[0];
            expect(
              await (result1 as GPUTensor<'float32'>)
                .copy()
                .compare(out, epsilon * 30)
            ).toBeTrue();
            const result2 = (await model.forward(inputsDevice))[0];
            expect(
              await (result2 as GPUTensor<'float32'>)
                .copy()
                .compare(out, epsilon * 30)
            ).toBeTrue();
          });
        }
      }
    });
  }
}
