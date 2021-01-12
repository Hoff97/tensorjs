import { CPUTensor } from "../../tensor/cpu/tensor";
import Tensor from "../../types";
import { OnnxNode } from "../node";
import { Attributes, Constants } from "../types";

export class SliceNode extends OnnxNode {
  private axes?: number[];
  private starts: number[];
  private ends: number[];

  constructor(attributes: Attributes, inputs: string[], outputs: string[], constants: Constants, onnxVersion: number) {
    super(attributes, inputs, outputs, constants, onnxVersion);

    this.axes = this.getAttributeInts("axes");
    this.starts = this.getAttributeInts("starts");
    this.ends = this.getAttributeInts("ends");
  }

  async forward(inputs: Tensor[]): Promise<Tensor[]> {
    if (this.onnxVersion < 11) {
      const a = inputs[0];

      return [a.slice(this.starts, this.ends, this.axes)];
    }
    throw new Error(`Slice not implemented for onnx version ${this.onnxVersion}`);
  }
}