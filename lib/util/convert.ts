import {Variable} from '../autograd/variable';
import {CPUTensor} from '../tensor/cpu/tensor';
import {GPUTensor} from '../tensor/gpu/tensor';
import {SparseTensor} from '../tensor/sparse/tensor';
import {WASMTensor} from '../tensor/wasm/tensor';
import Tensor, {DType} from '../types';

export type Backend = 'CPU' | 'WASM' | 'GPU';

/**
 * Convert a tensor to the specified backend
 */
export async function toBackend<DTpe extends DType>(
  tensor: Tensor<DTpe>,
  backend: Backend
): Promise<Tensor<DTpe>> {
  if (backend === 'CPU') {
    return toCPU(tensor);
  } else if (backend === 'WASM') {
    return toWASM(tensor);
  } else {
    return toGPU(tensor);
  }
}

export async function toCPU<DTpe extends DType>(
  tensor: Tensor<DTpe>
): Promise<Tensor<DTpe>> {
  if (tensor instanceof Variable) {
    return new Variable(await toCPU(tensor.value), {
      grad: tensor.grad !== undefined ? await toCPU(tensor.grad) : undefined,
    });
  } else if (tensor instanceof SparseTensor) {
    return new SparseTensor(
      await toCPU(tensor.values),
      await toCPU(tensor.indices),
      tensor.shape,
      tensor.denseDims
    );
  }
  if (tensor instanceof CPUTensor) {
    return tensor;
  }
  const values = await tensor.getValues();
  return new CPUTensor(tensor.getShape(), values, tensor.dtype);
}

export async function toWASM<DTpe extends DType>(
  tensor: Tensor<DTpe>
): Promise<Tensor<DTpe>> {
  if (tensor.dtype === 'float16') {
    throw new Error('Cant represent float16 tensor on Wasm backend');
  }
  if (tensor instanceof Variable) {
    return new Variable(await toWASM(tensor.value), {
      grad: tensor.grad !== undefined ? await toWASM(tensor.grad) : undefined,
    });
  } else if (tensor instanceof SparseTensor) {
    return new SparseTensor(
      await toWASM(tensor.values),
      await toWASM(tensor.indices),
      tensor.shape,
      tensor.denseDims
    );
  }
  if (tensor instanceof WASMTensor) {
    return tensor;
  }
  const values = await tensor.getValues();
  if (tensor instanceof CPUTensor && values instanceof Int32Array) {
    return tensor;
  }

  return new WASMTensor(
    Array.from(values),
    new Uint32Array(tensor.getShape()),
    tensor.dtype as any
  ) as Tensor<DTpe>;
}

export async function toGPU<DTpe extends DType>(
  tensor: Tensor<DTpe>
): Promise<Tensor<DTpe>> {
  if (tensor.dtype === 'float64') {
    throw new Error('Cant represent float64 tensor on WebGL backend');
  }
  if (tensor instanceof Variable) {
    return new Variable(await toGPU(tensor.value), {
      grad: tensor.grad !== undefined ? await toGPU(tensor.grad) : undefined,
    });
  } else if (tensor instanceof SparseTensor) {
    return new SparseTensor(
      await toGPU(tensor.values),
      await toGPU(tensor.indices),
      tensor.shape,
      tensor.denseDims
    );
  }
  if (tensor instanceof GPUTensor) {
    return tensor;
  }
  const values = await tensor.getValues();
  if (tensor instanceof CPUTensor && values instanceof Int32Array) {
    return tensor;
  }
  return new GPUTensor(
    Array.from(values),
    tensor.getShape(),
    tensor.dtype as any
  );
}

/**
 * Determines if the two tensors are of the same type, ie. if they are on the same backend
 */
export function sameType<DTpe1 extends DType, DTpe2 extends DType>(
  a: Tensor<DTpe1>,
  b: Tensor<DTpe2>
): boolean {
  if ((a.dtype as any) !== (b.dtype as any)) {
    return false;
  }
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
