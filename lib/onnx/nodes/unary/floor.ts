import { FloorOperation } from "../../../ops/gpu/floor";
import { UnaryOperation } from "../../../ops/gpu/unaryOperation";
import { gpuConstructor, GPUTensor } from "../../../tensor/gpu/tensor";
import Tensor from "../../../types";
import { Attributes, Constants } from "../../types";
import { UnaryNode } from "./unaryNode";

export class FloorNode extends UnaryNode {
  constructor(attributes: Attributes, inputs: string[], outputs: string[], constants: Constants, onnxVersion: number) {
    super(attributes, inputs, outputs, constants, onnxVersion);

    this.name = 'Floor';
  }

  compute(x: Tensor): Tensor {
    return x.floor();
  }

  getOperation(): UnaryOperation<GPUTensor> {
    return new FloorOperation(gpuConstructor, this.allocator);
  }
}