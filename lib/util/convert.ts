import { CPUTensor } from "../tensor/cpu/tensor";
import { GPUTensor } from "../tensor/gpu/tensor";
import { WASMTensor } from "../tensor/wasm/tensor";
import Tensor, { Precision }  from "../types";

export async function toCPU(tensor: Tensor) {
  if (tensor instanceof CPUTensor) {
    return tensor;
  }
  const values = await tensor.getValues();
  return new CPUTensor(tensor.getShape(), values);
}

export async function toWASM(tensor: Tensor) {
  if (tensor instanceof WASMTensor) {
    return tensor;
  }
  const values = await tensor.getValues();
  if (tensor instanceof CPUTensor && values instanceof Int32Array) {
    return tensor;
  }
  return new WASMTensor(values as Float32Array, new Uint32Array(tensor.getShape()));
}

export async function toGPU(tensor: Tensor, precision: Precision) {
  if (tensor instanceof GPUTensor) {
    return tensor;
  }
  const values = await tensor.getValues();
  if (tensor instanceof CPUTensor && values instanceof Int32Array) {
    return tensor;
  }
  return new GPUTensor(values as Float32Array, tensor.getShape(), precision);
}