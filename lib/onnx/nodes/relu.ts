import Tensor from "../../types";
import { OnnxNode } from "../node";
import { Attributes, Constants } from "../types";
import { ClipNode } from "./clip";

export class ReluNode extends ClipNode {
  constructor(attributes: Attributes, inputs: string[], outputs: string[], constants: Constants, onnxVersion: number) {
    super(attributes, inputs, outputs, constants, onnxVersion);

    this.max = undefined;
    this.min = 0;
  }

  getType() {
    return 'Relu';
  }
}