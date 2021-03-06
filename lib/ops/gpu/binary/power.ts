import {
  DTypeGpu,
  GPUTensorConstructor,
  GPUTensorI,
} from '../../../tensor/gpu/interface';
import {GPUMemoryAllocator} from '../../../tensor/gpu/memory';
import {BinaryOperation} from './binaryOperation';

export class PowerOperation<
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
    return `pow(${a}, ${b})`;
  }
}
