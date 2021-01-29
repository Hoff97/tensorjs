import { OnnxModel } from '../lib/onnx/model';
import { toGPU } from '../lib/util/convert';

import { onnx } from 'onnx-proto';
import { createTensor } from '../lib/onnx/util';
import Tensor, { Precision } from '../lib/types';
import { GPUTensor } from '../lib/tensor/gpu/tensor';

import { ModelData, models } from './data/models';
import { getSize } from '../lib/util/shape';
import { defaultAllocator } from '../lib/tensor/gpu/gl';

const epsilon = 0.1;

const modelArgs = {
  "ultraface": {
  },
  "mosaic": {
    noConvertNodes: [69, 98]
  }
}

const run = false;

function randomValues(length: number) {
  const values: number[] = [];
  for (let i = 0; i < length; i++) {
    values.push(Math.random());
  }
  return values;
}

async function loadData(modelData: ModelData, precision: Precision) {
  const resp = await fetch(`onnx/models/${modelData.name}/${modelData.name}.onnx`);
  const buffer = await resp.arrayBuffer();

  const args = {
  //@ts-ignore
    ...modelArgs[modelData.name],
    precision: precision
  };

  //@ts-ignore
  const model = new OnnxModel(buffer, args);

  const inputs: GPUTensor[] = [];
  let output: GPUTensor;

  if (modelData.inputData) {
    let i = 0;
    while (true) {
      const resp = await fetch(`onnx/models/${modelData.name}/test_data_set_0/input_${i}.pb`);
      if (resp.status !== 200) {
        break;
      }
      const buffer = await resp.arrayBuffer();
      const arr = new Uint8Array(buffer);
      const tensorProto = onnx.TensorProto.decode(arr);
      const tensor = await toGPU(createTensor(tensorProto), precision);
      inputs.push(tensor as GPUTensor);
      i++;
    }

    const respOut = await fetch(`onnx/models/${modelData.name}/test_data_set_0/output_0.pb`);
    const bufferOut = await respOut.arrayBuffer();
    const arr = new Uint8Array(bufferOut);
    const tensorProto = onnx.TensorProto.decode(arr);
    output = (await toGPU(createTensor(tensorProto), precision)) as GPUTensor;
  } else {
    for (let inputShape of modelData.inputsShape) {
      const values = randomValues(getSize(inputShape));
      inputs.push(new GPUTensor(new Float32Array(values), inputShape, precision));
    }
  }

  return {
    model, inputs, output
  };
}

function cleanup(model: OnnxModel, inputs: Tensor[], output?: Tensor) {
  model.delete();
  inputs.forEach(x => x.delete());
  if (output !== undefined) {
    output.delete();
  }
}

if (run) {
  for (let modelData of models) {
    describe(`Model ${modelData.name}`, () => {
      it(`Should work on GPU`, async () => {
        const { model, inputs, output } = await loadData(modelData, 32);

        model.allocator = defaultAllocator;

        await model.toGPU();

        const result1 = (await model.forward(inputs))[0];

        if (output !== undefined) {
          expect(await result1.compare(output, epsilon)).toBeTrue();
        }

        result1.delete();
        cleanup(model, inputs, output);
      });

      it(`Should work on GPU when precompiled`, async () => {
        const { model, inputs, output } = await loadData(modelData, 32);

        const modelCompiled = model;

        modelCompiled.allocator = defaultAllocator;

        await modelCompiled.toGPU();

        await modelCompiled.compileForInputs(inputs);
        const result1 = (await modelCompiled.forward(inputs))[0];

        if (output !== undefined) {
          expect(await result1.compare(output, epsilon)).toBeTrue();
        } else {
          const { model, inputs, output } = await loadData(modelData, 32);

          await model.toGPU();
          const result2 = (await model.forward(inputs))[0];
          expect(await result1.compare(result2, epsilon)).toBeTrue();

          cleanup(model, inputs, output);
        }

        result1.delete();
        cleanup(modelCompiled, inputs, output);
      });

      it(`Should work on GPU when optimized and compiled`, async () => {
        const { model, inputs, output } = await loadData(modelData, 32);

        const modelCompiled = model;

        modelCompiled.allocator = defaultAllocator;

        await modelCompiled.toGPU();

        modelCompiled.optimize();
        await modelCompiled.compileForInputs(inputs);
        const result1 = (await modelCompiled.forward(inputs))[0];

        if (output !== undefined) {
          expect(await result1.compare(output, epsilon)).toBeTrue();
        } else {
          const { model, inputs, output } = await loadData(modelData, 32);

          await model.toGPU();
          const result2 = (await model.forward(inputs))[0];
          expect(await result1.compare(result2, epsilon)).toBeTrue();

          cleanup(model, inputs, output);
        }

        result1.delete();
        cleanup(modelCompiled, inputs, output);
      });

      it(`Should work with half precision`, async () => {
        const { model, inputs, output } = await loadData(modelData, 16);

        const modelCompiled = model;

        modelCompiled.allocator = defaultAllocator;

        await modelCompiled.toGPU();

        modelCompiled.optimize();
        await modelCompiled.compileForInputs(inputs);
        const result1 = (await modelCompiled.forward(inputs))[0];

        if (output !== undefined) {
          expect(await result1.copy().compare(output.copy(), epsilon*2)).toBeTrue();
        } else {
          const { model, inputs, output } = await loadData(modelData, 32);

          await model.toGPU();
          const result2 = (await model.forward(inputs))[0];
          expect(await result1.copy().compare(result2.copy(), epsilon*2)).toBeTrue();

          cleanup(model, inputs, output);
        }

        result1.delete();
        cleanup(modelCompiled, inputs, output);
      });
    });
  }
}
