import { CPUTensor } from "../../tensor/cpu/tensor";
import Tensor from "../../types";
import { OnnxNode } from "../node";
import { Attributes, Constants } from "../types";
import { createTensor } from "../util";

export class ConstantOfShapeNode extends OnnxNode {
  private tensor: Tensor;

  constructor(attributes: Attributes, inputs: string[], outputs: string[], constants: Constants, onnxVersion: number) {
    super(attributes, inputs, outputs, constants, onnxVersion);

    if (onnxVersion < 11) {
      const tensor = this.getAttributeTensor("value");
      this.tensor = createTensor(tensor);
    }
  }

  forward(inputs: Tensor[]): Tensor[] {
    const _shape = inputs[0];

    if (!(_shape instanceof CPUTensor)) {
      throw new Error("Expand needs cpu tensor as shape tensor");
    }
    const shape = new Array(_shape.size);
    for (let i = 0; i < _shape.size; i++) {
      shape[i] = _shape.get(i);
    }

    console.log(shape);

    return [this.tensor.reshape(shape)];
  }
}