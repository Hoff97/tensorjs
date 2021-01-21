import { CeilOperation } from "../../../ops/gpu/ceil";
import { UnaryOperation } from "../../../ops/gpu/unaryOperation";
import { gpuConstructor, GPUTensor } from "../../../tensor/gpu/tensor";
import Tensor from "../../../types";
import { Attributes, Constants } from "../../types";
import { UnaryNode } from "./unaryNode";

export class CeilNode extends UnaryNode {
  constructor(attributes: Attributes, inputs: string[], outputs: string[], constants: Constants, onnxVersion: number) {
    super(attributes, inputs, outputs, constants, onnxVersion);
  }

  compute(x: Tensor): Tensor {
    return x.ceil();
  }

  getOperation(): UnaryOperation<GPUTensor> {
    return new CeilOperation(gpuConstructor, this.allocator);
  }
}