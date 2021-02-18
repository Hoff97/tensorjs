import {Mode} from '../../model/module';
import Tensor from '../../types';
import {OnnxNode} from '../node';
import {Attributes, Constants} from '../types';

export class SoftsignNode extends OnnxNode {
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

    const abs = x.abs();
    const p1 = abs.addScalar(1);
    abs.delete();
    const result = x.divide(p1);
    p1.delete();

    return [result];
  }

  getType() {
    return 'Softsign';
  }

  delete(): void {}
}
