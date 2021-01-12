import { CPUTensor } from "../../tensor/cpu/tensor";
import Tensor from "../../types";
import { OnnxNode } from "../node";
import { Attributes, Constants } from "../types";
import { createTensor } from "../util";

export class ExpandNode extends OnnxNode {
  constructor(attributes: Attributes, inputs: string[], outputs: string[], constants: Constants, onnxVersion: number) {
    super(attributes, inputs, outputs, constants, onnxVersion);
  }

  async forward(inputs: Tensor[]): Promise<Tensor[]> {
    if (this.onnxVersion < 13) {
      const tensor = inputs[0];

      const _shape = inputs[1];
      if (!(_shape instanceof CPUTensor)) {
        throw new Error("Expand needs cpu tensor as shape tensor");
      }
      const shape = new Array(_shape.size);
      for (let i = 0; i < _shape.size; i++) {
        shape[i] = _shape.get(i);
      }

      return [tensor.expand(shape)];
    }
    throw new Error('Expand not yet implemented');
  }
}