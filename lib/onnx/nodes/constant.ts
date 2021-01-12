import Tensor from "../../types";
import { OnnxNode } from "../node";
import { Attributes, Constants } from "../types";
import { createTensor } from "../util";

export class ConstantNode extends OnnxNode {
  private tensor: Tensor;

  constructor(attributes: Attributes, inputs: string[], outputs: string[], constants: Constants, onnxVersion: number) {
    super(attributes, inputs, outputs, constants, onnxVersion);

    if (onnxVersion < 11) {
      const tensor = this.getAttributeTensor("value");
      this.tensor = createTensor(tensor);
    }
  }

  async forward(inputs: Tensor[]): Promise<Tensor[]> {

    if (this.onnxVersion < 11) {
      return [this.tensor.copy()];
    }
    throw new Error('Constant with onnx version >= 11 not yet implemented');
  }
}