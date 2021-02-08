import {Variable} from '../autograd/variable';
import {CPUTensor} from '../tensor/cpu/tensor';
import {GPUTensor} from '../tensor/gpu/tensor';
import {WASMTensor} from '../tensor/wasm/tensor';
import Tensor, {Precision} from '../types';

export type Backend = 'CPU' | 'WASM' | 'GPU';

export async function toBackend(
  tensor: Tensor,
  backend: Backend,
  precision?: Precision
) {
  if (backend === 'CPU') {
    return toCPU(tensor);
  } else if (backend === 'WASM') {
    return toWASM(tensor);
  } else {
    if (precision === undefined) {
      precision = 32;
    }
    return toGPU(tensor, precision);
  }
}

export async function toCPU(tensor: Tensor): Promise<Tensor> {
  if (tensor instanceof Variable) {
    return new Variable(await toCPU(tensor.value), {
      grad: tensor.grad !== undefined ? await toCPU(tensor.grad) : undefined,
    });
  }
  if (tensor instanceof CPUTensor) {
    return tensor;
  }
  const values = await tensor.getValues();
  return new CPUTensor(tensor.getShape(), values);
}

export async function toWASM(tensor: Tensor): Promise<Tensor> {
  if (tensor instanceof Variable) {
    return new Variable(await toWASM(tensor.value), {
      grad: tensor.grad !== undefined ? await toWASM(tensor.grad) : undefined,
    });
  }
  if (tensor instanceof WASMTensor) {
    return tensor;
  }
  const values = await tensor.getValues();
  if (tensor instanceof CPUTensor && values instanceof Int32Array) {
    return tensor;
  }
  return new WASMTensor(
    values as Float32Array,
    new Uint32Array(tensor.getShape())
  );
}

export async function toGPU(
  tensor: Tensor,
  precision: Precision
): Promise<Tensor> {
  if (tensor instanceof Variable) {
    return new Variable(await toGPU(tensor.value, precision), {
      grad:
        tensor.grad !== undefined
          ? await toGPU(tensor.grad, precision)
          : undefined,
    });
  }
  if (tensor instanceof GPUTensor) {
    return tensor;
  }
  const values = await tensor.getValues();
  if (tensor instanceof CPUTensor && values instanceof Int32Array) {
    return tensor;
  }
  return new GPUTensor(values as Float32Array, tensor.getShape(), precision);
}

export function sameType(a: Tensor, b: Tensor): boolean {
  if (a instanceof Variable && b instanceof Variable) {
    return sameType(a.value, b.value);
  }
  if (a instanceof CPUTensor && b instanceof CPUTensor) {
    return true;
  }
  if (a instanceof WASMTensor && b instanceof WASMTensor) {
    return true;
  }
  if (a instanceof GPUTensor && b instanceof GPUTensor) {
    return true;
  }
  return false;
}
