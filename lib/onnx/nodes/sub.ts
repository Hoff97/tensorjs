import { BinaryOperation } from "../../ops/gpu/binaryOperation";
import { SubtractOperation } from "../../ops/gpu/subtract";
import { gpuConstructor, GPUTensor } from "../../tensor/gpu/tensor";
import Tensor from "../../types";
import { Attributes, Constants } from "../types";
import { BinaryNode } from "./binaryNode";

export class SubNode extends BinaryNode {
  constructor(attributes: Attributes, inputs: string[], outputs: string[], constants: Constants, onnxVersion: number) {
    super(attributes, inputs, outputs, constants, onnxVersion);

    this.name = 'Sub';
  }

  compute(a: Tensor, b: Tensor): Tensor {
    return a.subtract(b);
  }

  getOperation(): BinaryOperation<GPUTensor> {
    return new SubtractOperation(gpuConstructor, this.allocator);
  }
}