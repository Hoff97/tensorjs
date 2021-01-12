import { CPUTensor } from "../../tensor/cpu/tensor";
import Tensor from "../../types";
import { OnnxNode } from "../node";
import { Attributes, Constants } from "../types";

export class GatherNode extends OnnxNode {
  private axis: number;

  constructor(attributes: Attributes, inputs: string[], outputs: string[], constants: Constants, onnxVersion: number) {
    super(attributes, inputs, outputs, constants, onnxVersion);

    this.axis = this.getAttributeInt("axis") || 0;
  }

  async forward(inputs: Tensor[]): Promise<Tensor[]> {
    const x = inputs[0];
    const indices = inputs[1];

    if (!(indices instanceof CPUTensor)) {
      throw new Error("Gather requires CPU tensor for the indices");
    }

    return [x.gather(this.axis, indices)];
  }
}