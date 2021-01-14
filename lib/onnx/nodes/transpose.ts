import Tensor from "../../types";
import { OnnxNode } from "../node";
import { Attributes, Constants } from "../types";

export class TransposeNode extends OnnxNode {
  private permutation?: number[];
  constructor(attributes: Attributes, inputs: string[], outputs: string[], constants: Constants, onnxVersion: number) {
    super(attributes, inputs, outputs, constants, onnxVersion);

    this.permutation = this.getAttributeInts("perm");
  }

  async forward(inputs: Tensor[]): Promise<Tensor[]> {
    const a = inputs[0];

    return [a.transpose(this.permutation)];
  }
}