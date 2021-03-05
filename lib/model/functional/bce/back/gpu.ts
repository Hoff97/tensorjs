import {BinaryOperation} from '../../../../ops/gpu/binary/binaryOperation';
import {Dispatcher} from '../../../../ops/gpu/dispatcher';
import {
  DTypeGpu,
  GPUTensorConstructor,
  GPUTensorI,
} from '../../../../tensor/gpu/interface';
import {GPUMemoryAllocator} from '../../../../tensor/gpu/memory';
import {gpuConstructor} from '../../../../tensor/gpu/tensor';

export class BCEBackOperation<
  GPUTensor extends GPUTensorI
> extends BinaryOperation<GPUTensor> {
  constructor(
    tensorConstructor: GPUTensorConstructor<GPUTensor>,
    dtype: DTypeGpu,
    allocator?: GPUMemoryAllocator
  ) {
    super(tensorConstructor, dtype, allocator);
  }

  getOp(a: string, b: string): string {
    return `${b} == 1.0 ? -1.0/${a} : 1.0/(1.0-${a})`;
  }
}

export const defaultBCEBackD = new Dispatcher(
  (dtype: DTypeGpu) => new BCEBackOperation(gpuConstructor, dtype)
);
