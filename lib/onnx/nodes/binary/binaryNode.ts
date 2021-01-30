import { BinaryOperation } from "../../../ops/gpu/binary/binaryOperation";
import { PrototypeTensor } from "../../../tensor/cpu/prototype";
import { CPUTensor } from "../../../tensor/cpu/tensor";
import { GPUTensor } from "../../../tensor/gpu/tensor";
import Tensor, { Precision } from "../../../types";
import { getSize } from "../../../util/shape";
import { OnnxNode } from "../../node";
import { Attributes, Constants } from "../../types";

export abstract class BinaryNode extends OnnxNode {
  protected name: string;

  constructor(attributes: Attributes, inputs: string[], outputs: string[], constants: Constants, onnxVersion: number) {
    super(attributes, inputs, outputs, constants, onnxVersion);
  }

  abstract compute(a: Tensor, b: Tensor): Tensor;

  async forward(inputs: Tensor[]): Promise<Tensor[]> {
    if (this.onnxVersion < 13 && this.onnxVersion >= 7) {
      const a = inputs[0];
      const b = inputs[1];

      return [this.compute(a,b)];
    }
    throw new Error(`${this.name} not implemented for onnx version ${this.onnxVersion}`);
  }

  getType() {
    return this.name;
  }

  delete(): void {}
}