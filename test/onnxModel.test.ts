import {OnnxModel} from '../lib/onnx/model';
import {toGPU} from '../lib/util/convert';

import {onnx} from 'onnx-proto';
import {createTensor} from '../lib/onnx/util';
import Tensor from '../lib/types';
import {GPUTensor} from '../lib/tensor/gpu/tensor';

import {ModelData, models} from './data/models';
import {getSize} from '../lib/util/shape';

const epsilon = 0.1;

const modelArgs = {
  superresolution: {},
  ultraface: {},
  mosaic: {
    noConvertNodes: [69, 98],
  },
};

const run = false;

function randomValues(length: number) {
  const values: number[] = [];
  for (let i = 0; i < length; i++) {
    values.push(Math.random());
  }
  return values;
}

async function loadData(modelData: ModelData) {
  const resp = await fetch(
    `onnx/models/${modelData.name}/${modelData.name}.onnx`
  );
  const buffer = await resp.arrayBuffer();

  const args = {
    //@ts-ignore
    ...modelArgs[modelData.name],
  };

  //@ts-ignore
  const model = new OnnxModel(buffer, args);

  const inputs: GPUTensor<'float32'>[] = [];
  let output: GPUTensor<'float32'>;

  if (modelData.inputData) {
    let i = 0;
    // eslint-disable-next-line no-constant-condition
    while (true) {
      const resp = await fetch(
        `onnx/models/${modelData.name}/test_data_set_0/input_${i}.pb`
      );
      if (resp.status !== 200) {
        break;
      }
      const buffer = await resp.arrayBuffer();
      const arr = new Uint8Array(buffer);
      const tensorProto = onnx.TensorProto.decode(arr);
      const tensor = await toGPU(createTensor(tensorProto));
      inputs.push(tensor as GPUTensor<'float32'>);
      i++;
    }

    const respOut = await fetch(
      `onnx/models/${modelData.name}/test_data_set_0/output_0.pb`
    );
    const bufferOut = await respOut.arrayBuffer();
    const arr = new Uint8Array(bufferOut);
    const tensorProto = onnx.TensorProto.decode(arr);
    output = (await toGPU(createTensor(tensorProto))) as GPUTensor<'float32'>;
  } else {
    //@ts-ignore
    for (const inputShape of modelData.inputsShape) {
      const values = randomValues(getSize(inputShape));
      inputs.push(new GPUTensor(values, inputShape, 'float32'));
    }
  }

  return {
    model,
    inputs,
    //@ts-ignore
    output,
  };
}

function cleanup(
  model: OnnxModel,
  inputs: Tensor<any>[],
  output?: Tensor<any>
) {
  model.delete();
  inputs.forEach(x => x.delete());
  if (output !== undefined) {
    output.delete();
  }
}

if (run) {
  for (const modelData of models) {
    describe(`Model ${modelData.name}`, () => {
      it('Should work on GPU', async () => {
        const {model, inputs, output} = await loadData(modelData);

        await model.toGPU();

        const result1 = (await model.forward(inputs))[0];

        if (output !== undefined) {
          expect(await result1.compare(output, epsilon)).toBeTrue();
        }

        result1.delete();
        cleanup(model, inputs, output);
      });

      it('Should work on GPU when optimized', async () => {
        const {model, inputs, output} = await loadData(modelData);

        const modelCompiled = model;

        await modelCompiled.toGPU();

        modelCompiled.optimize();
        const result1 = (await modelCompiled.forward(inputs))[0];

        if (output !== undefined) {
          expect(await result1.compare(output, epsilon)).toBeTrue();
        } else {
          const {model, inputs, output} = await loadData(modelData);

          await model.toGPU();
          const result2 = (await model.forward(inputs))[0];
          expect(await result1.compare(result2, epsilon)).toBeTrue();

          cleanup(model, inputs, output);
        }

        result1.delete();
        cleanup(modelCompiled, inputs, output);
      });

      it('Should work with half precision', async () => {
        // TODO: Load models/data with half precision here
        const {model, inputs, output} = await loadData(modelData);

        const modelCompiled = model;

        await modelCompiled.toGPU();

        modelCompiled.optimize();
        const result1 = (await modelCompiled.forward(inputs))[0];

        if (output !== undefined) {
          expect(
            await (result1 as GPUTensor<'float32'>)
              .copy()
              .compare((output as GPUTensor<'float32'>).copy(), epsilon * 2)
          ).toBeTrue();
        } else {
          const {model, inputs, output} = await loadData(modelData);

          await model.toGPU();
          const result2 = (await model.forward(inputs))[0];
          expect(
            await (result1 as GPUTensor<'float32'>)
              .copy()
              .compare((result2 as GPUTensor<'float32'>).copy(), epsilon * 2)
          ).toBeTrue();

          cleanup(model, inputs, output);
        }

        result1.delete();
        cleanup(modelCompiled, inputs, output);
      });
    });
  }
}
