import {Mode} from '../../../model/module';
import Tensor, {DType} from '../../../types';
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
    name: string,
    mode: Mode
  ) {
    super(attributes, inputs, outputs, constants, onnxVersion, mode);

    this.name = name;
  }

  abstract compute<DTpe extends DType>(
    a: Tensor<DTpe>,
    b: Tensor<DTpe>
  ): Tensor<DTpe>;

  async forward<DTpe extends DType>(
    inputs: Tensor<DTpe>[]
  ): Promise<Tensor<DTpe>[]> {
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
