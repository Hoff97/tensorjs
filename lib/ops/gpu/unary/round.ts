import {
  DTypeGpu,
  GPUTensorConstructor,
  GPUTensorI,
} from '../../../tensor/gpu/interface';
import {GPUMemoryAllocator} from '../../../tensor/gpu/memory';
import {UnaryOperation} from './unaryOperation';

export class RoundOperation<
  GPUTensor extends GPUTensorI
> extends UnaryOperation<GPUTensor> {
  constructor(
    tensorConstructor: GPUTensorConstructor<GPUTensor>,
    dtype: DTypeGpu,
    allocator?: GPUMemoryAllocator
  ) {
    super(tensorConstructor, dtype, allocator);
  }

  operation(input: string): string {
    return `floor(${input}+vec4(0.5))`;
  }
}
