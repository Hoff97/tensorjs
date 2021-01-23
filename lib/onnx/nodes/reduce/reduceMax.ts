import { MaxOperation } from "../../../ops/gpu/max";
import { PoolOperation } from "../../../ops/gpu/pool";
import { gpuConstructor, GPUTensor } from "../../../tensor/gpu/tensor";
import Tensor from "../../../types";
import { OnnxNode } from "../../node";
import { Attributes, Constants } from "../../types";
import { ReduceNode } from "./reduceNode";

export class ReduceMaxNode extends ReduceNode {
  constructor(attributes: Attributes, inputs: string[], outputs: string[], constants: Constants, onnxVersion: number) {
    super(attributes, inputs, outputs, constants, onnxVersion);

    this.name = 'ReduceMax';
  }

  calc(input: Tensor): Tensor {
    return input.max(this.axes, this.keepDims);
  }

  getOperation(): PoolOperation<GPUTensor> {
    return new MaxOperation(gpuConstructor, this.allocator);
  }
}