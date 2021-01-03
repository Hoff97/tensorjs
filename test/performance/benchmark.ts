import CPUTensor from "../../js/tensor/cpu/tensor";
import GPUTensor from "../../js/tensor/gpu/tensor";
import WASMTensor, { wasmLoaded } from "../../js/tensor/wasm/tensor";
import Tensor from "../../js/types";

declare const suite: any;
declare const benchmark: any;

const cpuConstructor = (shape: ReadonlyArray<number>, values: number[]) => new CPUTensor(shape, values);
const gpuConstructor = (shape: ReadonlyArray<number>, values: number[]) => {
  const vals = Float32Array.from(values);
  return new GPUTensor(vals, shape);
};
const wasmConstructor = (shape: ReadonlyArray<number>, values: number[]) => {
  const sh = Uint32Array.from(shape);
  const vals = Float32Array.from(values);
  return new WASMTensor(vals, sh);
};

// TODO: Find a way to use wasm
const backends = [
  { name: 'CPU', constructor: cpuConstructor },
//  { name: 'WASM', constructor: wasmConstructor },
  { name: 'GPU', constructor: gpuConstructor },
];

function randomValues(length: number) {
  const values: number[] = [];
  for (let i = 0; i < length; i++) {
    values.push(Math.random());
  }
  return values;
}

suite("Tensor create", () => {
  const values = randomValues(100*100);
  for (let backend of backends) {
    benchmark(backend.name, () => {
      const tensor = backend.constructor([100,100], values);
      tensor.delete();
    });
  }
});

suite("Tensor exp", () => {
  const values = randomValues(100*100);

  const tensors: {[name: string]: Tensor} = {};
  for (let backend of backends) {
    tensors[backend.name] = backend.constructor([100,100], values);
  }

  for (let backend of backends) {
    benchmark(backend.name, () => {
      const result = tensors[backend.name].exp();
      result.delete();
    });
  }
});

suite("Tensor matmul", () => {
  const values1 = randomValues(100*100);
  const values2 = randomValues(100*100);

  const tensors: {[name: string]: Tensor[]} = {};
  for (let backend of backends) {
    tensors[backend.name] = [];
    tensors[backend.name].push(backend.constructor([100,100], values1));
    tensors[backend.name].push(backend.constructor([100,100], values2));
  }

  for (let backend of backends) {
    benchmark(backend.name, () => {
      const result = tensors[backend.name][0].matMul(tensors[backend.name][1]);
      result.delete();
    });
  }
});