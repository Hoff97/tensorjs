import Tensor from "../../../types";
import { OnnxNode } from "../../node";
import { Attributes, Constants } from "../../types";

export class ReduceSumNode extends OnnxNode {
  private axes?: number[];
  private keepDims?: boolean;

  constructor(attributes: Attributes, inputs: string[], outputs: string[], constants: Constants, onnxVersion: number) {
    super(attributes, inputs, outputs, constants, onnxVersion);

    this.axes = this.getAttributeInts("axes");
    const keep = this.getAttributeInt("keepdims");

    this.keepDims = keep === 1 || keep === undefined;
  }

  async forward(inputs: Tensor[]): Promise<Tensor[]> {
    if (this.onnxVersion < 11) {
      return [inputs[0].sum(this.axes, this.keepDims)];
    }
    throw new Error(`Reduce mean is not implemented for onnx version ${this.onnxVersion}`);
  }
}