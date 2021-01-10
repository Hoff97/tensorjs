import { CPUTensor } from "../tensor/cpu/tensor";
import { GPUTensor } from "../tensor/gpu/tensor";
import { WASMTensor } from "../tensor/wasm/tensor";
import Tensor  from "../types";

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
  if (tensor instanceof CPUTensor && tensor.type === "int") {
    return tensor;
  }
  const values = await tensor.getValues();
  return new WASMTensor(values, new Uint32Array(tensor.getShape()));
}

export async function toGPU(tensor: Tensor) {
  if (tensor instanceof GPUTensor) {
    return tensor;
  }
  if (tensor instanceof CPUTensor && tensor.type === "int") {
    return tensor;
  }
  const values = await tensor.getValues();
  return new GPUTensor(values, tensor.getShape());
}