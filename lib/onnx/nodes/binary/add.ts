import { AddOperation } from "../../../ops/gpu/binary/add";
import { BinaryOperation } from "../../../ops/gpu/binary/binaryOperation";
import { gpuConstructor, GPUTensor } from "../../../tensor/gpu/tensor";
import Tensor from "../../../types";
import { Attributes, Constants } from "../../types";
import { BinaryNode } from "./binaryNode";

export class AddNode extends BinaryNode {
  constructor(attributes: Attributes, inputs: string[], outputs: string[], constants: Constants, onnxVersion: number) {
    super(attributes, inputs, outputs, constants, onnxVersion);

    this.name = 'Add';
  }

  compute(a: Tensor, b: Tensor): Tensor {
    return a.add(b);
  }

  getOperation(): BinaryOperation<GPUTensor> {
    return new AddOperation(gpuConstructor, this.allocator);
  }
}