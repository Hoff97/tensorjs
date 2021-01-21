import { BinaryOperation } from "../../ops/gpu/binaryOperation";
import { MultiplyOperation } from "../../ops/gpu/multiply";
import { gpuConstructor, GPUTensor } from "../../tensor/gpu/tensor";
import types from "../../types";
import { Attributes, Constants } from "../types";
import { BinaryNode } from "./binaryNode";

export class MulNode extends BinaryNode {
  constructor(attributes: Attributes, inputs: string[], outputs: string[], constants: Constants, onnxVersion: number) {
    super(attributes, inputs, outputs, constants, onnxVersion);

    this.name = 'Mul';
  }

  compute(a: types, b: types): types {
    return a.multiply(b);
  }

  getOperation(): BinaryOperation<GPUTensor> {
    return new MultiplyOperation(gpuConstructor, this.allocator);
  }
}