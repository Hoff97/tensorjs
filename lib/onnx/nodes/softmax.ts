import Tensor from "../../types";
import { OnnxNode } from "../node";
import { Attributes, Constants } from "../types";

export class SoftmaxNode extends OnnxNode {
  private axis?: number;

  constructor(attributes: Attributes, inputs: string[], outputs: string[], constants: Constants, onnxVersion: number) {
    super(attributes, inputs, outputs, constants, onnxVersion);

    this.axis = this.getAttributeInt("axis");
  }

  async forward(inputs: Tensor[]): Promise<Tensor[]> {
    const x = inputs[0];

    const shapeX = x.getShape();

    let ax = this.axis;
    if (ax === undefined) {
      if (this.onnxVersion < 13) {
        ax = 1;
      } else {
        ax = shapeX.length - 1;
      }
    }

    const sh1 = shapeX.slice(0, ax).reduce((x,y) => x*y, 1);

    const reshaped = x.reshape([sh1,-1], false);

    const max = reshaped.max(1, true);
    const normalized = reshaped.subtract(max);
    const exp = normalized.exp();
    const sum = exp.sum(1, true);
    const result = exp.divide(sum);

    max.delete();
    normalized.delete();
    exp.delete();
    sum.delete();

    return [result.reshape(shapeX, false)];
  }
}