import { PoolOperation } from "../../../ops/gpu/pool/pool";
import { SumOperation } from "../../../ops/gpu/pool/sum";
import { gpuConstructor, GPUTensor } from "../../../tensor/gpu/tensor";
import Tensor from "../../../types";
import { Attributes, Constants } from "../../types";
import { ReduceNode } from "./reduceNode";

export class ReduceSumNode extends ReduceNode {
  constructor(attributes: Attributes, inputs: string[], outputs: string[], constants: Constants, onnxVersion: number) {
    super(attributes, inputs, outputs, constants, onnxVersion);

    this.name = 'ReduceSum';
  }

  calc(input: Tensor): Tensor {
    return input.sum(this.axes, this.keepDims);
  }
}