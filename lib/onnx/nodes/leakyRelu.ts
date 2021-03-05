import {Mode} from '../../model/module';
import Tensor, {DType} from '../../types';
import {OnnxNode} from '../node';
import {Attributes, Constants} from '../types';

export class LeakyReluNode extends OnnxNode {
  alpha: number;

  constructor(
    attributes: Attributes,
    inputs: string[],
    outputs: string[],
    constants: Constants,
    onnxVersion: number,
    mode: Mode
  ) {
    super(attributes, inputs, outputs, constants, onnxVersion, mode);

    this.alpha = this.getAttributeFloat('alpha') || 0.01;
  }

  async forward<DTpe extends DType>(
    inputs: Tensor<DTpe>[]
  ): Promise<Tensor<DTpe>[]> {
    const tensor = inputs[0];

    const below = tensor.clip(undefined, 0);
    const above = tensor.clip(0, undefined);

    const result = below.add(above, this.alpha);

    below.delete();
    above.delete();
    return [result];
  }

  delete(): void {}

  getType() {
    return 'LeakyRelu';
  }
}
