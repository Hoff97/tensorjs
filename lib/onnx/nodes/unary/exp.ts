import { ExpOperation } from "../../../ops/gpu/exp";
import { UnaryOperation } from "../../../ops/gpu/unaryOperation";
import { gpuConstructor, GPUTensor } from "../../../tensor/gpu/tensor";
import Tensor from "../../../types";
import { Attributes, Constants } from "../../types";
import { UnaryNode } from "./unaryNode";

export class ExpNode extends UnaryNode {
  constructor(attributes: Attributes, inputs: string[], outputs: string[], constants: Constants, onnxVersion: number) {
    super(attributes, inputs, outputs, constants, onnxVersion);

    this.name = 'Exp';
  }

  compute(x: Tensor): Tensor {
    return x.exp();
  }

  getOperation(): UnaryOperation<GPUTensor> {
    return new ExpOperation(gpuConstructor, this.allocator);
  }
}