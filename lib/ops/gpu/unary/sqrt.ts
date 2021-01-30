import { GPUTensorConstructor, GPUTensorI } from "../../../tensor/gpu/interface";
import { GPUMemoryAllocator } from "../../../tensor/gpu/memory";
import { UnaryOperation, UnaryOpInfo, UnaryOpInput } from "./unaryOperation";


export class SqrtOperation<GPUTensor extends GPUTensorI> extends UnaryOperation<GPUTensor> {
  constructor(tensorConstructor: GPUTensorConstructor<GPUTensor>, allocator?: GPUMemoryAllocator) {
    super(tensorConstructor, allocator);
  }

  operation(input: string): string {
    return `sqrt(${input})`;
  }
}
