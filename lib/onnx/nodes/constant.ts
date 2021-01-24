import { PrototypeTensor } from "../../tensor/cpu/prototype";
import { CPUTensor } from "../../tensor/cpu/tensor";
import Tensor from "../../types";
import { toCPU, toGPU, toWASM } from "../../util/convert";
import { OnnxNode } from "../node";
import { Attributes, Constants } from "../types";
import { createTensor } from "../util";

export class ConstantNode extends OnnxNode {
  public tensor: Tensor;

  constructor(attributes: Attributes, inputs: string[], outputs: string[], constants: Constants, onnxVersion: number) {
    super(attributes, inputs, outputs, constants, onnxVersion);

    if (onnxVersion < 11) {
      const tensor = this.getAttributeTensor("value");
      this.tensor = createTensor(tensor);
    }
  }

  async forward(inputs: Tensor[]): Promise<Tensor[]> {
    if (this.onnxVersion < 11) {
      return [this.tensor];
    }
    throw new Error('Constant with onnx version >= 11 not yet implemented');
  }

  async staticForward(inputs: Tensor[], compile: boolean): Promise<{ outputs: (CPUTensor | PrototypeTensor)[]; }> {
    if (this.onnxVersion < 11) {
      return { outputs: [this.tensor as any] };
    }
    throw new Error('Constant with onnx version >= 11 not yet implemented');
  }

  initializeForCompiling(): void {
  }

  async toCPU() {
    this.tensor = await toCPU(this.tensor);
  }

  async toWASM() {
    this.tensor = await toWASM(this.tensor);
  }
  async toGPU() {
    this.tensor = await toGPU(this.tensor);
  }
}