import { PrototypeTensor } from "../../tensor/cpu/prototype";
import { CPUTensor } from "../../tensor/cpu/tensor";
import Tensor from "../../types";
import { getSize } from "../../util/shape";
import { OnnxNode } from "../node";
import { Attributes, Constants } from "../types";
import { createTensor } from "../util";

export class ConstantOfShapeNode extends OnnxNode {
  private tensor: CPUTensor;

  constructor(attributes: Attributes, inputs: string[], outputs: string[], constants: Constants, onnxVersion: number) {
    super(attributes, inputs, outputs, constants, onnxVersion);

    if (onnxVersion < 11) {
      const tensor = this.getAttributeTensor("value");
      this.tensor = createTensor(tensor);
    }
  }

  async forward(inputs: Tensor[]): Promise<Tensor[]> {
    const _shape = inputs[0];

    if (!(_shape instanceof CPUTensor)) {
      throw new Error("ConstantOfShape needs cpu tensor as shape tensor");
    }
    const shape = new Array(_shape.size);
    for (let i = 0; i < _shape.size; i++) {
      shape[i] = _shape.get(i);
    }

    const size = getSize(shape);
    const values = new Float32Array(size).fill(this.tensor.get(0));

    return [new CPUTensor(shape, values, this.tensor.type)];
  }

  staticForward(inputs: Tensor[], compile: boolean): Promise<{ outputs: (CPUTensor | PrototypeTensor)[]; }> {
    return this.defaultStaticForward(inputs);
  }

  initializeForCompiling(): void {
  }

  getType() {
    return 'ConstantOfShape';
  }

  delete(): void {
    this.tensor.delete();
  }
}