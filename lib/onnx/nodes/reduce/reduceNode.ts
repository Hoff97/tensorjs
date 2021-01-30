import { Tensor } from "../../../library";
import { PoolOperation } from "../../../ops/gpu/pool/pool";
import { PrototypeTensor } from "../../../tensor/cpu/prototype";
import { CPUTensor } from "../../../tensor/cpu/tensor";
import { GPUTensor } from "../../../tensor/gpu/tensor";
import types, { Precision } from "../../../types";
import { getSize } from "../../../util/shape";
import { OnnxNode } from "../../node";
import { Attributes, Constants } from "../../types";

export abstract class ReduceNode extends OnnxNode {
  protected axes?: number[];
  protected keepDims?: boolean;

  protected name: string;

  constructor(attributes: Attributes, inputs: string[], outputs: string[], constants: Constants, onnxVersion: number) {
    super(attributes, inputs, outputs, constants, onnxVersion);

    this.axes = this.getAttributeInts("axes");
    const keep = this.getAttributeInt("keepdims");

    this.keepDims = keep === 1 || keep === undefined;
  }

  abstract calc(input: Tensor): Tensor;

  protected getAxes(input: Tensor) {
    if (this.axes !== undefined) {
      return this.axes;
    } else {
      const rank = input.getShape().length;

      const res = new Array(rank);
      for (let i = 0; i < rank; i++) {
        res[i] = i;
      }
      return res;
    }
  }

  async forward(inputs: types[]): Promise<types[]> {
    if (this.onnxVersion < 11) {

      return [this.calc(inputs[0])];
    }
    throw new Error(`${this.name} is not implemented for onnx version ${this.onnxVersion}`);
  }

  getType() {
    return this.name;
  }

  delete(): void {}
}