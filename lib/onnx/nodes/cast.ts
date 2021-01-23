import { cast } from "../../ops/cpu/cast";
import { PrototypeTensor } from "../../tensor/cpu/prototype";
import { CPUTensor } from "../../tensor/cpu/tensor";
import { glContext } from "../../tensor/gpu/gl";
import Tensor from "../../types";
import { toCPU, toGPU, toWASM } from "../../util/convert";
import { OnnxNode } from "../node";
import { Attributes, Constants } from "../types";

export class CastNode extends OnnxNode {
  private to: string;

  constructor(attributes: Attributes, inputs: string[], outputs: string[], constants: Constants, onnxVersion: number) {
    super(attributes, inputs, outputs, constants, onnxVersion);

    this.to = this.getAttributeString("to");
  }

  async forward(inputs: Tensor[]): Promise<Tensor[]> {
    const x = inputs[0];

    if (x instanceof CPUTensor) {
      return [cast(x, this.to)];
    }
    throw new Error("Can only cast CPU tensors right now");
  }

  staticForward(inputs: Tensor[], compile: boolean): Promise<{ outputs: (CPUTensor | PrototypeTensor)[]; }> {
    if (this.allStaticCPU(inputs)) {
      return this.defaultStaticForward(inputs);
    }

    throw new Error("Can only cast CPU tensors right now");
  }

  initializeForCompiling(): void {
    throw new Error("Method not implemented.");
  }
}