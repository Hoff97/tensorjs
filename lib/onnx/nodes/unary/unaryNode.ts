import Tensor from "../../../types";
import { OnnxNode } from "../../node";
import { Attributes, Constants } from "../../types";

export abstract class UnaryNode extends OnnxNode {
  protected name: string;

  constructor(attributes: Attributes, inputs: string[], outputs: string[], constants: Constants, onnxVersion: number) {
    super(attributes, inputs, outputs, constants, onnxVersion);
  }

  abstract compute(x: Tensor): Tensor;

  async forward(inputs: Tensor[]): Promise<Tensor[]> {
    const x = inputs[0];

    return [this.compute(x)];
  }

  getType() {
    return this.name;
  }

  delete(): void {}
}