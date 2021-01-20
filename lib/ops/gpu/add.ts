import { GPUTensorConstructor, GPUTensorI } from "../../tensor/gpu/interface";
import { GPUMemoryAllocator } from "../../tensor/gpu/memory";
import { BinaryOperation } from "./binaryOperation";

export class AddOperation<GPUTensor extends GPUTensorI> extends BinaryOperation<GPUTensor> {
  constructor(tensorConstructor: GPUTensorConstructor<GPUTensor>, allocator?: GPUMemoryAllocator) {
    super(tensorConstructor, allocator);
  }

  getOp(a: string, b: string): string {
    return `${a} + ${b}`;
  }
}
