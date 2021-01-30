import { PoolOperation } from "../../../ops/gpu/pool/pool";
import { SumSquareOperation } from "../../../ops/gpu/pool/sumSquare";
import { gpuConstructor, GPUTensor } from "../../../tensor/gpu/tensor";
import types from "../../../types";
import { Attributes, Constants } from "../../types";
import { ReduceNode } from "./reduceNode";

export class ReduceSumSquareNode extends ReduceNode {
  constructor(attributes: Attributes, inputs: string[], outputs: string[], constants: Constants, onnxVersion: number) {
    super(attributes, inputs, outputs, constants, onnxVersion);

    this.name = 'ReduceSumSquare';
  }

  calc(input: types): types {
    return input.sumSquare(this.axes, this.keepDims);
  }
}