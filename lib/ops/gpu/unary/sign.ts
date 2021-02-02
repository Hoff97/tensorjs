import {GPUTensorConstructor, GPUTensorI} from '../../../tensor/gpu/interface';
import {GPUMemoryAllocator} from '../../../tensor/gpu/memory';
import {UnaryOperation} from './unaryOperation';

export class SignOperation<
  GPUTensor extends GPUTensorI
> extends UnaryOperation<GPUTensor> {
  constructor(
    tensorConstructor: GPUTensorConstructor<GPUTensor>,
    allocator?: GPUMemoryAllocator
  ) {
    super(tensorConstructor, allocator);
  }

  operation(input: string): string {
    return `${input} < 0.0 ? -1.0 : 1.0`;
  }
}
