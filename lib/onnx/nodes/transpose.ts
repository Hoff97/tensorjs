import {Mode} from '../../model/module';
import Tensor, {DType} from '../../types';
import {OnnxNode} from '../node';
import {Attributes, Constants} from '../types';

export class TransposeNode extends OnnxNode {
  private permutation?: number[];

  constructor(
    attributes: Attributes,
    inputs: string[],
    outputs: string[],
    constants: Constants,
    onnxVersion: number,
    mode: Mode
  ) {
    super(attributes, inputs, outputs, constants, onnxVersion, mode);

    this.permutation = this.getAttributeInts('perm');
  }

  async forward<DTpe extends DType>(
    inputs: Tensor<DTpe>[]
  ): Promise<Tensor<DTpe>[]> {
    const a = inputs[0];

    return [a.transpose(this.permutation)];
  }

  getType() {
    return 'Transpose';
  }

  delete(): void {}
}
