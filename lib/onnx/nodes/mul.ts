import { GPUTensor } from "../../tensor/gpu/tensor";
import Tensor from "../../types";
import { toGPU } from "../../util/convert";
import { OnnxNode } from "../node";
import { Attributes, Constants } from "../types";

export class MulNode extends OnnxNode {
  constructor(attributes: Attributes, inputs: string[], outputs: string[], constants: Constants, onnxVersion: number) {
    super(attributes, inputs, outputs, constants, onnxVersion);
  }

  async forward(inputs: Tensor[]): Promise<Tensor[]> {
    if (this.onnxVersion < 13 && this.onnxVersion >= 7) {
      const a = inputs[0];
      const b = inputs[1];

      return [a.multiply(b)];
    }
    throw new Error(`Add not implemented for onnx version ${this.onnxVersion}`);
  }
}