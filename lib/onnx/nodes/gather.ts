import { GatherInfo, GatherOperation } from "../../ops/gpu/util/gather";
import { PrototypeTensor } from "../../tensor/cpu/prototype";
import { CPUTensor } from "../../tensor/cpu/tensor";
import { gpuConstructor, GPUTensor } from "../../tensor/gpu/tensor";
import Tensor, { Precision } from "../../types";
import { getSize } from "../../util/shape";
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

  getType() {
    return 'Gather';
  }

  delete(): void {}
}