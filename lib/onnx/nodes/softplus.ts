import {Mode} from '../../model/module';
import Tensor from '../../types';
import {OnnxNode} from '../node';
import {Attributes, Constants} from '../types';

export class SoftplusNode extends OnnxNode {
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

  async forward(inputs: Tensor[]): Promise<Tensor[]> {
    const x = inputs[0];

    const exp = x.exp();
    const p1 = exp.addScalar(1);
    exp.delete();
    const result = p1.log();
    p1.delete();

    return [result];
  }

  getType() {
    return 'Softplus';
  }

  delete(): void {}
}
