import { GPUTensorConstructor, GPUTensorI } from "../../../tensor/gpu/interface";
import { GPUMemoryAllocator } from "../../../tensor/gpu/memory";
import { Precision } from "../../../types";
import { UnaryOperation } from "./unaryOperation";

export class AbsOperation<GPUTensor extends GPUTensorI> extends UnaryOperation<GPUTensor> {
  constructor(tensorConstructor: GPUTensorConstructor<GPUTensor>, allocator?: GPUMemoryAllocator) {
    super(tensorConstructor, allocator);
  }

  operation(input: string): string {
    return `abs(${input})`;
  }
}
