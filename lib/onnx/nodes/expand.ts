import { ExpandInfo, ExpandOperation } from "../../ops/gpu/util/expand";
import { PrototypeTensor } from "../../tensor/cpu/prototype";
import { CPUTensor } from "../../tensor/cpu/tensor";
import { gpuConstructor, GPUTensor } from "../../tensor/gpu/tensor";
import Tensor, { Precision } from "../../types";
import { getSize } from "../../util/shape";
import { OnnxNode } from "../node";
import { Attributes, Constants } from "../types";
import { createTensor } from "../util";

export class ExpandNode extends OnnxNode {
  constructor(attributes: Attributes, inputs: string[], outputs: string[], constants: Constants, onnxVersion: number) {
    super(attributes, inputs, outputs, constants, onnxVersion);
  }

  async forward(inputs: Tensor[]): Promise<Tensor[]> {
    if (this.onnxVersion < 13) {
      let tensor = inputs[0];

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
    throw new Error(`Expand not yet implemented for onnx version ${this.onnxVersion}`);
  }

  getType() {
    return 'Expand';
  }

  delete(): void {}
}