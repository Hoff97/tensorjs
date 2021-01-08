import Tensor from "../../types";
import { OnnxNode } from "../node";
import { Attributes, Constants } from "../types";

export class ConcatNode extends OnnxNode {
  private axis: number;

  constructor(attributes: Attributes, inputs: string[], outputs: string[], constants: Constants, onnxVersion: number) {
    super(attributes, inputs, outputs, constants, onnxVersion);

    if (onnxVersion < 13) {
      this.axis = this.getAttributeInt("axis");
    }
  }

  forward(inputs: Tensor[]): Tensor[] {
    if (inputs.length > 2) {
      throw new Error("Concat with more than 2 tensors not yet supported");
    }

    const a = inputs[0];
    const b = inputs[1];

    return [a.concat(b, this.axis)];
  }
}