import Tensor from "../../types";
import { OnnxNode } from "../node";
import { Attributes, Constants } from "../types";

export class FloorNode extends OnnxNode {
  constructor(attributes: Attributes, inputs: string[], outputs: string[], constants: Constants, onnxVersion: number) {
    super(attributes, inputs, outputs, constants, onnxVersion);
  }

  async forward(inputs: Tensor[]): Promise<Tensor[]> {
    const a = inputs[0];

    return [a.floor()];
  }
}