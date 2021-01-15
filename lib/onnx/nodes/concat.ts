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

  async forward(inputs: Tensor[]): Promise<Tensor[]> {
    if (inputs.length > 2) {
      console.warn(`Concat with more than 2 tensors is currently slow. Doing concat with ${inputs.length} tensors`);
    }

    let result = inputs[0];
    for (let i = 1; i < inputs.length; i++) {
      let newRes = result.concat(inputs[i], this.axis);
      if (i > 1) {
        result.delete();
      }
      result = newRes;
    }

    return [result];
  }
}