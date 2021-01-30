import { PoolOperation } from "../../../ops/gpu/pool/pool";
import { ReduceMeanOperation } from "../../../ops/gpu/pool/reduceMean";
import { gpuConstructor, GPUTensor } from "../../../tensor/gpu/tensor";
import Tensor from "../../../types";
import { OnnxNode } from "../../node";
import { Attributes, Constants } from "../../types";
import { ReduceNode } from "./reduceNode";

export class ReduceMeanNode extends ReduceNode {
  constructor(attributes: Attributes, inputs: string[], outputs: string[], constants: Constants, onnxVersion: number) {
    super(attributes, inputs, outputs, constants, onnxVersion);

    this.name = 'ReduceMean';
  }

  calc(input: Tensor): Tensor {
    return input.reduceMean(this.axes, this.keepDims);
  }
}