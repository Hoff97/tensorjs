import Tensor from '../../../types';
import {OnnxNode} from '../../node';
import {Attributes, Constants} from '../../types';

export abstract class BinaryNode extends OnnxNode {
  protected name: string;

  constructor(
    attributes: Attributes,
    inputs: string[],
    outputs: string[],
    constants: Constants,
    onnxVersion: number,
    name: string
  ) {
    super(attributes, inputs, outputs, constants, onnxVersion);

    this.name = name;
  }

  abstract compute(a: Tensor, b: Tensor): Tensor;

  async forward(inputs: Tensor[]): Promise<Tensor[]> {
    if (this.onnxVersion < 13 && this.onnxVersion >= 7) {
      const a = inputs[0];
      const b = inputs[1];

      return [this.compute(a, b)];
    }
    throw new Error(
      `${this.name} not implemented for onnx version ${this.onnxVersion}`
    );
  }

  getType() {
    return this.name;
  }

  delete(): void {}
}
