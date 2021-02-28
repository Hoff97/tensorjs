import {Mode} from '../../model/module';
import Tensor, {DType} from '../../types';
import {OnnxNode} from '../node';
import {Attributes, Constants} from '../types';

export class SqueezeNode extends OnnxNode {
  constructor(
    attributes: Attributes,
    inputs: string[],
    outputs: string[],
    constants: Constants,
    onnxVersion: number,
    mode: Mode
  ) {
    super(attributes, inputs, outputs, constants, onnxVersion, mode);
  }

  async forward<DTpe extends DType>(
    inputs: Tensor<DTpe>[]
  ): Promise<Tensor<DTpe>[]> {
    const tensor = inputs[0];

    return [tensor.squeeze()];
  }

  getType() {
    return 'Squeeze';
  }

  delete(): void {}
}
