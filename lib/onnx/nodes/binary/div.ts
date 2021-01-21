import { BinaryOperation } from "../../../ops/gpu/binaryOperation";
import { DivideOperation } from "../../../ops/gpu/divide";
import { gpuConstructor, GPUTensor } from "../../../tensor/gpu/tensor";
import Tensor from "../../../types";
import { Attributes, Constants } from "../../types";
import { BinaryNode } from "./binaryNode";

export class DivNode extends BinaryNode {
  constructor(attributes: Attributes, inputs: string[], outputs: string[], constants: Constants, onnxVersion: number) {
    super(attributes, inputs, outputs, constants, onnxVersion);

    this.name = 'Div';
  }

  compute(a: Tensor, b: Tensor): Tensor {
    return a.divide(b);
  }

  getOperation(): BinaryOperation<GPUTensor> {
    return new DivideOperation(gpuConstructor, this.allocator);
  }
}